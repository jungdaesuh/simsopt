"""Reference-lane optimizer wrappers and SciPy host adapters.

SciPy's public ``minimize()`` contract is host-array based: ``x0`` is an
``ndarray`` and, when ``jac=True``, the objective callback returns
``(float, array_like)``. This module is therefore the intentional host NumPy
boundary for the CPU/reference lane. JAX target execution must stay on the
``target_*`` entrypoints in ``optimizer_jax.py`` and must never route through
these helpers.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from . import optimizer_jax as _optimizer

__all__ = [
    "reference_least_squares",
    "reference_minimize",
    "_scipy_dispatch",
    "_scipy_minimize",
    "_scipy_minimize_value_and_grad",
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


def _scipy_dispatch(scipy_fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_dispatch",
        method=method,
    )
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
        # to the reference-only adapter instead of the JAX target lane.
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
):
    """Run the CPU/reference optimizer lane."""
    if method not in _optimizer._REFERENCE_METHODS:
        raise ValueError(
            "reference_minimize() only supports SciPy reference methods "
            f"{sorted(_optimizer._REFERENCE_METHODS)}. Got {method!r}."
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
