"""Reference-lane optimizer wrappers and SciPy host adapters.

SciPy's public ``minimize()`` contract is host-array based: ``x0`` is an
``ndarray`` and, when ``jac=True``, the objective callback returns
``(float, array_like)``. This module is therefore the intentional host NumPy
boundary for SciPy-controlled lanes. CPU/reference execution enters through
``reference_*`` helpers; the explicit target SciPy-control lane enters through
``target_scipy_minimize_value_and_grad`` from ``optimizer_jax.target_minimize``.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from .optimizer_host_lbfgs import (
    host_invalid_step_log_to_list,
    lbfgs_status_is_success,
    lbfgs_status_message,
    minimize_lbfgs_host_core,
)
from . import optimizer_jax as _optimizer

__all__ = [
    "reference_least_squares",
    "reference_minimize",
    "_scipy_dispatch",
    "_scipy_minimize",
    "_scipy_minimize_value_and_grad",
    "target_scipy_minimize_value_and_grad",
]


def _strip_internal_options(options, method):
    if not options:
        return {}
    internal = {
        "line_search_maxiter",
        "callback",
        "progress_callback",
    }
    if method == "bfgs":
        internal |= {"maxcor", "ftol", "maxfun", "maxgrad", "maxls"}
    elif method == "lbfgs":
        internal |= {"maxgrad"}
    return {key: value for key, value in options.items() if key not in internal}


def _normalize_scipy_result(result, *, x_dtype):
    result.x = _optimizer._optimizer_flat_vector(result.x, dtype=x_dtype)
    result.jac = _optimizer._optimizer_flat_vector(result.jac, dtype=x_dtype)
    result.nit = int(getattr(result, "nit", 0))
    result.nfev = int(getattr(result, "nfev", 0))
    if hasattr(result, "njev"):
        result.njev = int(result.njev)
    result.success = bool(result.success)
    if hasattr(result, "status"):
        result.status = int(result.status)
    return result


def _scipy_dispatch_core(scipy_fun, x0, *, method, tol, maxiter, options):
    stripped_options = _strip_internal_options(options, method)
    x_dtype = _optimizer._optimizer_dtype(x0)
    if method == "bfgs":
        scipy_method = "BFGS"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, **stripped_options}
    else:
        scipy_method = "L-BFGS-B"
        scipy_opts = {
            "maxiter": maxiter,
            "gtol": tol,
            "maxcor": 200,
            **stripped_options,
        }
    callback = options.get("callback")
    scipy_callback = None
    if callback is not None:

        def scipy_callback(x_np):
            callback(_optimizer._optimizer_flat_vector(x_np, dtype=x_dtype))

    return _normalize_scipy_result(
        # SciPy consumes host arrays here by contract; keep that cast confined
        # to the explicit SciPy-control adapter.
        scipy_minimize(
            scipy_fun,
            np.asarray(x0, dtype=np.dtype(x_dtype)),
            jac=True,
            method=scipy_method,
            options=scipy_opts,
            callback=scipy_callback,
        ),
        x_dtype=x_dtype,
    )


def _scipy_dispatch(scipy_fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_dispatch",
        method=method,
    )
    return _scipy_dispatch_core(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _scipy_minimize(fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_minimize",
        method=method,
    )
    val_and_grad_fn = _optimizer._cached_jit_value_and_grad(fun)
    x_dtype = _optimizer._optimizer_dtype(x0)

    def scipy_fun(x_np):
        x_jax = _optimizer._optimizer_flat_vector(x_np, dtype=x_dtype)
        val, grad = val_and_grad_fn(x_jax)
        # ``minimize(jac=True)`` expects ``(float, array_like)`` on the host.
        return float(val), np.asarray(grad, dtype=np.dtype(x_dtype))

    return _scipy_dispatch(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def target_scipy_minimize_value_and_grad(
    fun,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
):
    """Run SciPy L-BFGS-B control against a JAX target value/grad evaluator."""
    if method != "lbfgs":
        raise ValueError(
            "target_scipy_minimize_value_and_grad() only supports method='lbfgs'."
        )
    x_dtype = _optimizer._optimizer_dtype(x0)

    def scipy_fun(x_np):
        x_jax = _optimizer._optimizer_flat_vector(x_np, dtype=x_dtype)
        val, grad = fun(x_jax)
        # ``minimize(jac=True)`` expects ``(float, array_like)`` on the host.
        return float(val), np.asarray(grad, dtype=np.dtype(x_dtype))

    return _scipy_dispatch_core(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _scipy_minimize_value_and_grad(fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_minimize_value_and_grad",
        method=method,
    )
    x_dtype = _optimizer._optimizer_dtype(x0)

    def scipy_fun(x_np):
        x_jax = _optimizer._optimizer_flat_vector(x_np, dtype=x_dtype)
        val, grad = fun(x_jax)
        # ``minimize(jac=True)`` expects ``(float, array_like)`` on the host.
        return float(val), np.asarray(grad, dtype=np.dtype(x_dtype))

    return _scipy_dispatch(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _host_trace_result_to_optimize_result(result):
    invalid_state = (not np.isfinite(result.f_k)) or (
        not np.all(np.isfinite(result.g_k))
    )
    invalid_step_log = host_invalid_step_log_to_list(result.invalid_step_events)
    return OptimizeResult(
        x=np.asarray(result.x_k),
        fun=float(result.f_k),
        jac=np.asarray(result.g_k),
        nit=int(result.k),
        nfev=int(result.nfev),
        njev=int(result.ngev),
        success=lbfgs_status_is_success(result.status, invalid_state),
        status=int(result.status),
        message=lbfgs_status_message(result.status, invalid_state),
        ls_status=int(result.ls_status),
        line_search_final_status=int(result.ls_status),
        maxiter_hit=int(result.status) == 1,
        rejected_step_count=len(invalid_step_log),
        invalid_step_log=invalid_step_log,
        optimizer_state_trace=tuple(result.optimizer_state_trace),
    )


def _trace_minimize_value_and_grad(
    fun,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
    initial_value_and_grad=None,
):
    _optimizer._require_native_cpu_reference_backend_for_trace_adapter(
        component="optimizer_jax_reference._trace_minimize_value_and_grad",
        method=method,
    )
    if method != "lbfgs-trace":
        raise ValueError(f"Unknown CPU/C++ trace optimizer method {method!r}.")
    x_dtype = _optimizer._optimizer_dtype(x0)

    def eval_value_and_grad_host(x_np):
        x_host = np.asarray(x_np, dtype=np.dtype(x_dtype))
        val, grad = fun(x_host)
        return float(val), np.asarray(grad, dtype=np.dtype(x_dtype))

    result = minimize_lbfgs_host_core(
        eval_value_and_grad_host,
        np.asarray(x0, dtype=np.dtype(x_dtype)),
        maxiter=maxiter,
        gtol=tol,
        maxcor=int(options.get("maxcor", 200)),
        ftol=float(options.get("ftol", tol)),
        maxfun=options.get("maxfun"),
        maxgrad=options.get("maxgrad"),
        maxls=int(options.get("maxls", 20)),
        initial_step_size=options.get("initial_step_size"),
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
        failure_callback=options.get("failure_callback"),
        initial_value_and_grad=initial_value_and_grad,
    )
    return _host_trace_result_to_optimize_result(result)


def reference_least_squares(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Run the CPU/reference least-squares lane."""
    if method != "lm":
        raise ValueError(
            "reference_least_squares() only supports method='lm'. "
            f"Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    _optimizer._raise_if_target_lane_required(
        component="optimizer_jax_reference.reference_least_squares",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
    )
    _optimizer._raise_if_strict_optimizer_fallback(
        component="optimizer_jax_reference.reference_least_squares",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
    )
    result = _optimizer.levenberg_marquardt(
        residual_fn,
        x0,
        maxiter=maxiter,
        tol=tol,
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
    )

    nit = int(_optimizer._host_scalar(result["nit"], dtype=np.int64))
    status = int(_optimizer._host_scalar(result["status"], dtype=np.int64))
    success = _optimizer._host_bool(result["success"])
    return OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        residual=result["residual"],
        residual_jacobian=result["residual_jacobian"],
        hessian=result["hessian"],
        damping=result["damping"],
        nit=nit,
        nfev=nit + 1,
        njev=nit + 1,
        status=status,
        success=success,
        message=_optimizer._least_squares_result_message(
            status,
            success,
        ),
    )


def reference_minimize(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
):
    """Run the CPU/reference optimizer lane."""
    if method in _optimizer._REFERENCE_TRACE_METHODS and not value_and_grad:
        raise ValueError(
            "reference_minimize() requires value_and_grad=True for "
            "method='lbfgs-trace'."
        )
    if method not in _optimizer._REFERENCE_METHODS | _optimizer._REFERENCE_TRACE_METHODS:
        raise ValueError(
            "reference_minimize() only supports reference methods "
            f"{sorted(_optimizer._REFERENCE_METHODS | _optimizer._REFERENCE_TRACE_METHODS)}. "
            f"Got {method!r}."
        )

    fun, x0, callback, pytree_adapter = _optimizer._prepare_optimizer_callable_inputs(
        fun,
        x0,
        value_and_grad=value_and_grad,
        callback=callback,
    )

    def finalize(result):
        return _optimizer._finalize_optimizer_result(result, pytree_adapter)

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    if failure_callback is not None:
        options["failure_callback"] = failure_callback

    _optimizer._raise_if_target_lane_required(
        component="optimizer_jax_reference.reference_minimize",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_OPTIMIZER_DETAIL,
    )
    _optimizer._raise_if_strict_optimizer_fallback(
        component="optimizer_jax_reference.reference_minimize",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_OPTIMIZER_DETAIL,
    )

    if method in _optimizer._REFERENCE_TRACE_METHODS:
        return finalize(
            _trace_minimize_value_and_grad(
                fun,
                x0,
                method=method,
                tol=tol,
                maxiter=maxiter,
                options=options,
                initial_value_and_grad=initial_value_and_grad,
            )
        )

    scipy_adapter = (
        _scipy_minimize_value_and_grad if value_and_grad else _scipy_minimize
    )
    return finalize(
        scipy_adapter(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
        )
    )
