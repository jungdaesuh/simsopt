"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.

Least-squares methods:
  - ``method="lm"``: host-driven Levenberg-Marquardt for residual-vector
    objectives on the reference lane.
  - ``method="lm-ondevice"``: trace-safe Levenberg-Marquardt for
    residual-vector objectives on the target lane.

Transitional private method (validated on JAX 0.9.2):
  - ``method="bfgs-hybrid"``: SciPy BFGS prefix, then JAX on-device BFGS.

Target private methods (validated on JAX 0.9.2):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: JAX on-device L-BFGS.

The private methods live in ``optimizer_jax_private/`` and intentionally mirror
the JAX 0.9.2 optimizer semantics so the line-search and iteration behavior
stay stable across this project. The reference source is the upstream
``jax-v0.9.2`` tag (``a659757d768587a81d095a9fab5f0c36f8beb218``).

This module contains zero ``jax._src`` imports. The private package now does as
well; both paths use public JAX APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import lax
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from ..backend import raise_if_strict_jax_fallback

__all__ = [
    "ContinuousOptimizerContract",
    "PRIVATE_OPTIMIZER_JAX_VERSION",
    "VALID_LEAST_SQUARES_ALGORITHMS",
    "VALID_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "jax_least_squares",
    "jax_minimize",
    "levenberg_marquardt",
    "levenberg_marquardt_traceable",
    "newton_polish",
    "newton_polish_traceable",
    "newton_exact",
    "newton_exact_traceable",
    "require_target_backend_x64",
    "resolve_least_squares_optimizer_method",
    "resolve_continuous_optimizer_contract",
    "resolve_optimizer_backend_method",
    "resolve_outer_loop_optimizer_contract",
]


PRIVATE_OPTIMIZER_JAX_VERSION = "0.9.2"
VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "hybrid", "ondevice"})
OPTIMIZER_BACKEND_ROLE = {
    "scipy": "reference",
    "hybrid": "transitional",
    "ondevice": "target",
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = frozenset({"hybrid", "ondevice"})
VALID_LEAST_SQUARES_ALGORITHMS = frozenset({"quasi-newton", "lm"})
_SUPPORTED_METHODS = {
    "bfgs",
    "lbfgs",
    "bfgs-hybrid",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
_SUPPORTED_LEAST_SQUARES_METHODS = frozenset({"lm", "lm-ondevice"})
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_REFERENCE_LEAST_SQUARES_DETAIL = (
    "the host-side reference least-squares optimizer lane"
)
_STRICT_HYBRID_OPTIMIZER_DETAIL = "the transitional SciPy-prefix hybrid optimizer lane"


@dataclass(frozen=True)
class ContinuousOptimizerContract:
    method: str
    use_scalar_objective: bool


@dataclass(frozen=True)
class _OptimizerPytreeAdapter:
    flat_x0: jax.Array
    unravel: Callable[[jax.Array], object]
    tree_def: object

    def wrap_fun(self, fun, *, value_and_grad: bool):
        if not value_and_grad:
            def wrapped(flat_x):
                return fun(self.unravel(jnp.asarray(flat_x)))

            return wrapped

        def wrapped(flat_x):
            flat_x = jnp.asarray(flat_x)
            value, grad_tree = fun(self.unravel(flat_x))
            _, grad_tree_def = jax.tree_util.tree_flatten(grad_tree)
            if grad_tree_def != self.tree_def:
                raise ValueError(
                    "Explicit value-and-gradient objectives must return a gradient "
                    "with the same pytree structure as x0."
                )
            flat_grad, _ = ravel_pytree(grad_tree)
            flat_grad = jnp.asarray(flat_grad, dtype=flat_x.dtype)
            if flat_grad.shape != flat_x.shape:
                raise ValueError(
                    "Explicit value-and-gradient objectives must return a gradient "
                    f"matching the flattened x0 shape {flat_x.shape}, got {flat_grad.shape}."
                )
            return jnp.asarray(value, dtype=flat_x.dtype), flat_grad

        return wrapped

    def wrap_callback(self, callback):
        if callback is None:
            return None

        def wrapped(flat_x):
            callback(_hostify_optimizer_tree(self.unravel(jnp.asarray(flat_x))))

        return wrapped

    def finalize_result(self, result):
        if hasattr(result, "x"):
            result.x = _hostify_optimizer_tree(self.unravel(jnp.asarray(result.x)))
        if hasattr(result, "jac"):
            result.jac = _hostify_optimizer_tree(
                self.unravel(jnp.asarray(result.jac))
            )
        return result


def _raise_if_strict_optimizer_fallback(*, method: str, detail: str) -> None:
    raise_if_strict_jax_fallback(
        component="optimizer_jax.jax_minimize",
        detail=f"{detail} for method={method!r}",
    )


def _x64_enabled():
    return bool(jax.config.jax_enable_x64)


def _device_scalar(value, *, dtype=jnp.float64):
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _hostify_optimizer_leaf(leaf):
    if isinstance(leaf, jax.Array):
        array = np.asarray(jax.device_get(leaf))
    elif isinstance(leaf, (np.ndarray, np.generic)) or np.isscalar(leaf):
        array = np.asarray(leaf)
    else:
        return leaf
    if array.ndim == 0:
        return array.item()
    return array


def _hostify_optimizer_tree(value):
    return jax.tree_util.tree_map(_hostify_optimizer_leaf, value)


def _is_flat_optimizer_vector(x0) -> bool:
    if isinstance(x0, (jax.Array, np.ndarray)):
        return x0.ndim == 1
    if isinstance(x0, (list, tuple)):
        try:
            array = np.asarray(x0)
        except Exception:
            return False
        return array.dtype != object and array.ndim == 1
    return False


def _prepare_optimizer_pytree_adapter(x0):
    if _is_flat_optimizer_vector(x0):
        return None
    flat_x0, unravel = ravel_pytree(x0)
    _, tree_def = jax.tree_util.tree_flatten(x0)
    return _OptimizerPytreeAdapter(
        flat_x0=jnp.asarray(flat_x0),
        unravel=unravel,
        tree_def=tree_def,
    )


def _prepare_optimizer_callable_inputs(fun, x0, *, value_and_grad, callback):
    adapter = _prepare_optimizer_pytree_adapter(x0)
    if adapter is None:
        return fun, x0, callback, None
    return (
        adapter.wrap_fun(fun, value_and_grad=value_and_grad),
        adapter.flat_x0,
        adapter.wrap_callback(callback),
        adapter,
    )


def _finalize_optimizer_result(result, adapter):
    if adapter is None:
        return result
    return adapter.finalize_result(result)


# ---------------------------------------------------------------------------
# Private package — lazy, one-way (private imports only constants defined above).
# The package is loaded on first access to a private symbol, so importing
# optimizer_jax for SciPy / Newton paths never touches private optimizer internals.
# ---------------------------------------------------------------------------
_private_pkg = None  # None = untried, False = absent, module = loaded

_PRIVATE_LAZY_NAMES = frozenset(
    {
        "_BFGSResults",
        "_line_search",
        "_line_search_value_and_grad",
        "_make_bfgs_continuation_state",
        "_minimize_bfgs_private",
        "_minimize_lbfgs_private",
        "_minimize_lbfgs_private_value_and_grad",
        "_private_bfgs_result_to_optimize_result",
        "_private_lbfgs_result_to_optimize_result",
        "_scipy_result_is_continuable",
    }
)


def _load_private_pkg():
    global _private_pkg
    if _private_pkg is None:
        try:
            from . import optimizer_jax_private

            _private_pkg = optimizer_jax_private
        except ImportError:
            _private_pkg = False
    return _private_pkg


def __getattr__(name):
    if name in _PRIVATE_LAZY_NAMES:
        pkg = _load_private_pkg()
        if pkg is False:
            raise AttributeError(
                f"Private optimizer symbol {name!r} requires the private package "
                f"(simsopt.geo.optimizer_jax_private). Install with: pip install -e ."
            )
        val = getattr(pkg, name)
        globals()[name] = val  # cache for subsequent access / monkeypatch
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_optimizer_backend_method(optimizer_backend, *, limited_memory):
    """Map the public backend contract to the concrete optimizer method."""
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, hybrid, ondevice.")
    if optimizer_backend == "scipy":
        return "lbfgs" if limited_memory else "bfgs"
    if optimizer_backend == "hybrid":
        if limited_memory:
            raise ValueError(
                "optimizer_backend='hybrid' is transitional and does not support "
                "limited_memory=True."
            )
        return "bfgs-hybrid"
    return "lbfgs-ondevice" if limited_memory else "bfgs-ondevice"


def resolve_least_squares_optimizer_method(
    optimizer_backend,
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Map the LS backend contract to the concrete least-squares method."""
    if least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS:
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(
            f"least_squares_algorithm must be one of: {allowed}."
        )
    if least_squares_algorithm == "quasi-newton":
        return resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=limited_memory,
        )
    if limited_memory:
        raise ValueError(
            "least_squares_algorithm='lm' is incompatible with limited_memory=True."
        )
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, hybrid, ondevice.")
    if optimizer_backend == "hybrid":
        raise ValueError(
            "least_squares_algorithm='lm' is unsupported for "
            "optimizer_backend='hybrid'."
        )
    return "lm" if optimizer_backend == "scipy" else "lm-ondevice"


def require_target_backend_x64(optimizer_backend):
    """Fail fast when a target-lane backend is requested without float64."""
    if optimizer_backend not in TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS:
        return
    if _x64_enabled():
        return
    role = OPTIMIZER_BACKEND_ROLE[optimizer_backend]
    raise RuntimeError(
        f"optimizer_backend='{optimizer_backend}' ({role}) requires "
        "jax_enable_x64=True before import/use."
    )


def resolve_continuous_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    limited_memory,
    allow_hybrid,
    component_label,
):
    """Resolve the shared continuous-optimizer route for outer solve lanes."""
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, hybrid, ondevice.")
    if field_backend != "jax" and optimizer_backend != "scipy":
        raise ValueError(
            f"{component_label} CPU/reference lane only supports "
            "optimizer_backend='scipy'."
        )
    if optimizer_backend == "hybrid":
        if not allow_hybrid:
            raise ValueError(
                "optimizer_backend='hybrid' is transitional and not supported for "
                f"{component_label}."
            )
        limited_memory = False
    if (
        field_backend == "jax"
        and optimizer_backend in TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS
    ):
        require_target_backend_x64(optimizer_backend)
    return ContinuousOptimizerContract(
        method=resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=limited_memory,
        ),
        use_scalar_objective=field_backend == "jax" and optimizer_backend == "ondevice",
    )


def resolve_outer_loop_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    component_label,
):
    """Resolve the optimizer contract for an outer optimization loop.

    Shared by both Stage 2 and single-stage outer loops, which use
    ``limited_memory=True`` and ``allow_hybrid=False``.
    """
    return resolve_continuous_optimizer_contract(
        field_backend,
        optimizer_backend,
        limited_memory=True,
        allow_hybrid=False,
        component_label=component_label,
    )


# ---------------------------------------------------------------------------
# SciPy adapter helpers
# ---------------------------------------------------------------------------


def _strip_internal_options(options, method):
    if not options:
        return {}
    internal = {
        "hybrid_scipy_maxiter",
        "line_search_maxiter",
        "callback",
        "progress_callback",
    }
    if method == "bfgs":
        internal |= {"maxcor", "ftol", "maxfun", "maxgrad", "maxls"}
    elif method == "lbfgs":
        internal |= {"maxgrad"}
    return {key: value for key, value in options.items() if key not in internal}


def _normalize_scipy_result(result):
    result.x = jnp.asarray(result.x)
    result.jac = jnp.asarray(result.jac)
    result.nit = int(getattr(result, "nit", 0))
    result.nfev = int(getattr(result, "nfev", 0))
    if hasattr(result, "njev"):
        result.njev = int(result.njev)
    result.success = bool(result.success)
    if hasattr(result, "status"):
        result.status = int(result.status)
    return result


def _scipy_dispatch(scipy_fun, x0, *, method, tol, maxiter, options):
    stripped_options = _strip_internal_options(options, method)
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
    return _normalize_scipy_result(
        scipy_minimize(
            scipy_fun,
            np.asarray(x0),
            jac=True,
            method=scipy_method,
            options=scipy_opts,
            callback=options.get("callback"),
        )
    )


def _scipy_minimize(fun, x0, *, method, tol, maxiter, options):
    val_and_grad_fn = jax.jit(jax.value_and_grad(fun))

    def scipy_fun(x_np):
        x_jax = jnp.asarray(x_np)
        val, grad = val_and_grad_fn(x_jax)
        return float(val), np.asarray(grad)

    return _scipy_dispatch(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _scipy_minimize_value_and_grad(fun, x0, *, method, tol, maxiter, options):
    def scipy_fun(x_np):
        val, grad = fun(np.asarray(x_np))
        return float(val), np.asarray(grad, dtype=float)

    return _scipy_dispatch(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _least_squares_cost(residual):
    residual = jnp.ravel(jnp.asarray(residual))
    return _device_scalar(0.5, dtype=residual.dtype) * jnp.vdot(residual, residual).real


def _least_squares_linearization(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    return gradient, hessian


def _clip_lm_damping(damping, *, dtype):
    minimum = _device_scalar(1.0e-12, dtype=dtype)
    maximum = _device_scalar(1.0e12, dtype=dtype)
    return jnp.clip(jnp.asarray(damping, dtype=dtype), minimum, maximum)


def _lm_defaults(dtype):
    return {
        "initial_damping": _device_scalar(1.0e-3, dtype=dtype),
        "accept_threshold": _device_scalar(0.0, dtype=dtype),
        "expand_factor": _device_scalar(4.0, dtype=dtype),
        "shrink_factor": _device_scalar(0.5, dtype=dtype),
        "mild_shrink_factor": _device_scalar(0.8, dtype=dtype),
        "ratio_low": _device_scalar(0.25, dtype=dtype),
        "ratio_high": _device_scalar(0.75, dtype=dtype),
        "predicted_floor": _device_scalar(1.0e-18, dtype=dtype),
    }


def _lm_iteration(
    residual_fn,
    jacobian_fn,
    state,
    *,
    tol,
):
    defaults = _lm_defaults(state["x"].dtype)
    damping = _clip_lm_damping(state["damping"], dtype=state["x"].dtype)
    damped_hessian = state["hessian"] + damping * jnp.eye(
        state["hessian"].shape[0],
        dtype=state["hessian"].dtype,
    )
    step = jnp.linalg.solve(damped_hessian, state["grad"])
    x_candidate = state["x"] - step
    residual_candidate = residual_fn(x_candidate)
    jacobian_candidate = jacobian_fn(x_candidate)
    cost_candidate = _least_squares_cost(residual_candidate)
    grad_candidate, hessian_candidate = _least_squares_linearization(
        residual_candidate,
        jacobian_candidate,
    )
    grad_norm_candidate = jnp.linalg.norm(grad_candidate, ord=jnp.inf)

    predicted_reduction = _device_scalar(0.5, dtype=state["x"].dtype) * jnp.dot(
        step,
        damping * step + state["grad"],
    )
    actual_reduction = state["cost"] - cost_candidate
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        jnp.all(jnp.isfinite(x_candidate))
        & jnp.all(jnp.isfinite(residual_candidate))
        & jnp.all(jnp.isfinite(jacobian_candidate))
        & jnp.isfinite(cost_candidate)
        & jnp.all(jnp.isfinite(grad_candidate))
        & jnp.all(jnp.isfinite(hessian_candidate))
    )
    accepted = finite_candidate & (actual_reduction > defaults["accept_threshold"])

    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["shrink_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["expand_factor"],
            lambda __: damping * defaults["mild_shrink_factor"],
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=state["x"].dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["expand_factor"],
            dtype=state["x"].dtype,
        ),
        operand=None,
    )

    return {
        "x": lax.select(accepted, x_candidate, state["x"]),
        "residual": lax.select(accepted, residual_candidate, state["residual"]),
        "residual_jacobian": lax.select(
            accepted,
            jacobian_candidate,
            state["residual_jacobian"],
        ),
        "cost": lax.select(accepted, cost_candidate, state["cost"]),
        "grad": lax.select(accepted, grad_candidate, state["grad"]),
        "hessian": lax.select(accepted, hessian_candidate, state["hessian"]),
        "grad_norm_inf": lax.select(
            accepted,
            grad_norm_candidate,
            state["grad_norm_inf"],
        ),
        "damping": next_damping,
        "nit": state["nit"] + 1,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "accepted": accepted,
        "success": finite_candidate & (grad_norm_candidate <= tol),
    }


def _least_squares_result_message(status, success):
    if bool(success):
        return "converged"
    if int(status) == 2:
        return "non-finite residual, jacobian, or step encountered"
    return "maximum iterations reached"


def levenberg_marquardt(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    callback=None,
    progress_callback=None,
):
    """Host-driven Levenberg-Marquardt solver for least-squares residuals."""
    residual_eval = jax.jit(residual_fn)
    jacobian_eval = jax.jit(jax.jacfwd(residual_fn))

    x = jnp.asarray(x0)
    residual = residual_eval(x)
    residual_jacobian = jacobian_eval(x)
    cost = _least_squares_cost(residual)
    grad, hessian = _least_squares_linearization(residual, residual_jacobian)
    grad_norm_inf = jnp.linalg.norm(grad, ord=jnp.inf)
    damping = _lm_defaults(x.dtype)["initial_damping"]
    status = 1
    success = bool(grad_norm_inf <= tol)
    nit = 0

    while nit < maxiter and not success:
        step_state = _lm_iteration(
            residual_eval,
            jacobian_eval,
            {
                "x": x,
                "residual": residual,
                "residual_jacobian": residual_jacobian,
                "cost": cost,
                "grad": grad,
                "hessian": hessian,
                "grad_norm_inf": grad_norm_inf,
                "damping": damping,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
                "status": jnp.asarray(status, dtype=jnp.int32),
                "accepted": jnp.asarray(False),
                "success": jnp.asarray(False),
            },
            tol=jnp.asarray(tol, dtype=x.dtype),
        )
        nit = int(step_state["nit"])
        status = int(step_state["status"])
        damping = step_state["damping"]
        if bool(step_state["accepted"]):
            x = step_state["x"]
            residual = step_state["residual"]
            residual_jacobian = step_state["residual_jacobian"]
            cost = step_state["cost"]
            grad = step_state["grad"]
            hessian = step_state["hessian"]
            grad_norm_inf = step_state["grad_norm_inf"]
            if callback is not None:
                callback(x)
            if progress_callback is not None:
                progress_callback(nit, float(cost), float(grad_norm_inf))
        success = bool(step_state["success"])
        if status == 2:
            break

    return {
        "x": x,
        "residual": residual,
        "residual_jacobian": residual_jacobian,
        "fun": cost,
        "grad": grad,
        "hessian": hessian,
        "damping": damping,
        "nit": nit,
        "status": status,
        "success": success,
    }


def levenberg_marquardt_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    callback=None,
    progress_callback=None,
):
    """Trace-safe Levenberg-Marquardt solver for least-squares residuals."""
    residual_eval = residual_fn
    jacobian_eval = jax.jacfwd(residual_fn)
    tol_value = _device_scalar(tol, dtype=jnp.asarray(x0).dtype)

    def run_solver(x_init):
        residual0 = residual_eval(x_init)
        residual_jacobian0 = jacobian_eval(x_init)
        cost0 = _least_squares_cost(residual0)
        grad0, hessian0 = _least_squares_linearization(residual0, residual_jacobian0)
        grad_norm_inf0 = jnp.linalg.norm(grad0, ord=jnp.inf)
        state0 = {
            "x": x_init,
            "residual": residual0,
            "residual_jacobian": residual_jacobian0,
            "cost": cost0,
            "grad": grad0,
            "hessian": hessian0,
            "grad_norm_inf": grad_norm_inf0,
            "damping": _lm_defaults(x_init.dtype)["initial_damping"],
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "accepted": jnp.asarray(False),
            "success": grad_norm_inf0 <= tol_value,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (~state["success"])
                & (state["status"] != 2)
            )

        def body_fun(state):
            next_state = _lm_iteration(
                residual_eval,
                jacobian_eval,
                state,
                tol=tol_value,
            )
            if callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(callback, next_state["x"]),
                    lambda _: None,
                    operand=None,
                )
            if progress_callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        progress_callback,
                        next_state["nit"],
                        next_state["cost"],
                        next_state["grad_norm_inf"],
                    ),
                    lambda _: None,
                    operand=None,
                )
            return next_state

        state = lax.while_loop(cond_fun, body_fun, state0)
        return {
            "x": state["x"],
            "residual": state["residual"],
            "residual_jacobian": state["residual_jacobian"],
            "fun": state["cost"],
            "grad": state["grad"],
            "hessian": state["hessian"],
            "damping": state["damping"],
            "nit": state["nit"],
            "status": state["status"],
            "success": state["success"],
        }

    return jax.jit(run_solver)(x0)


# ---------------------------------------------------------------------------
# Newton solvers (public path, no jax._src)
# ---------------------------------------------------------------------------


def _hessian_vector_product_fn(objective_fn):
    grad_fn = jax.grad(objective_fn)
    return jax.jit(lambda x, v: jax.jvp(grad_fn, (x,), (v,))[1])


def _materialize_dense_hessian(hvp_fn, x):
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    cols = lax.map(lambda basis: hvp_fn(x, basis), eye)
    dense = jnp.swapaxes(cols, 0, 1)
    return 0.5 * (dense + dense.T)


def _stabilize_dense_hessian(H, stab):
    stab_value = jnp.asarray(stab, dtype=H.dtype)
    return H + stab_value * jnp.eye(H.shape[0], dtype=H.dtype)


def _newton_candidate_status(current_norm, x_next, grad_next):
    candidate_norm = jnp.linalg.norm(grad_next)
    accepted = (
        jnp.all(jnp.isfinite(x_next))
        & jnp.all(jnp.isfinite(grad_next))
        & (candidate_norm <= current_norm)
    )
    return accepted, candidate_norm


def _gmres_solve_newton_system(hvp_fn, x, rhs, *, stab, tol):
    n = rhs.shape[0]
    restart = max(5, min(n, 50))
    maxiter = max(10, min(4 * n, 200))
    stab_value = jnp.asarray(stab, dtype=rhs.dtype)

    def matvec(v):
        return hvp_fn(x, v) + stab_value * v

    dx, _ = gmres(
        matvec,
        rhs,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
    )
    residual = rhs - matvec(dx)
    return dx, residual, matvec


def newton_polish(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    progress_callback=None,
):
    """Newton polish using exact Hessian-vector products.

    Iterations solve the Newton system with GMRES against the exact
    Hessian linear operator, avoiding the peak memory cost of
    ``jax.hessian(objective_fn)`` on large Boozer LS problems.

    The dense Hessian is still materialized once at the final iterate so
    callers retain the existing adjoint/PLU contract.
    """
    val_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))
    hvp_fn = _hessian_vector_product_fn(objective_fn)

    x = x0
    val, grad = val_and_grad_fn(x)
    norm = jnp.linalg.norm(grad)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))
        dx, linear_residual, _ = _gmres_solve_newton_system(
            hvp_fn,
            x,
            grad,
            stab=stab,
            tol=linear_tol,
        )
        linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
        used_dense_fallback = False
        if not np.all(np.isfinite(np.asarray(dx))) or (
            linear_residual_norm > max(1e-10, 1e-3 * float(norm))
        ):
            H_solve = _stabilize_dense_hessian(
                _materialize_dense_hessian(hvp_fn, x),
                stab,
            )
            dx = jnp.linalg.solve(H_solve, grad)
            linear_residual = grad - H_solve @ dx
            linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
            used_dense_fallback = True
        if (not used_dense_fallback) and linear_residual_norm > linear_tol:
            correction, _, _ = _gmres_solve_newton_system(
                hvp_fn,
                x,
                linear_residual,
                stab=stab,
                tol=linear_tol,
            )
            if np.all(np.isfinite(np.asarray(correction))):
                dx = dx + correction
        candidate_x = x - dx
        candidate_val, candidate_grad = val_and_grad_fn(candidate_x)
        accepted, candidate_norm = _newton_candidate_status(
            norm, candidate_x, candidate_grad
        )
        if not bool(accepted):
            break
        x = candidate_x
        val = candidate_val
        grad = candidate_grad
        norm = candidate_norm
        nit += 1
        if progress_callback is not None:
            progress_callback(nit, float(val), float(norm))

    H = _stabilize_dense_hessian(_materialize_dense_hessian(hvp_fn, x), stab)

    return {
        "x": x,
        "fun": val,
        "grad": grad,
        "hessian": H,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }


def newton_polish_traceable(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    progress_callback=None,
):
    """Trace-safe Newton polish for JAX-traceable objective paths.

    This variant keeps all loop state and fallback decisions inside JAX control
    flow so higher-level traced objectives can invoke the Newton stage without
    crossing back into Python.
    """
    val_and_grad_fn = jax.value_and_grad(objective_fn)
    hvp_fn = _hessian_vector_product_fn(objective_fn)
    tol_value = _device_scalar(tol)
    linear_tol = jnp.minimum(
        _device_scalar(1e-10),
        jnp.maximum(tol_value * _device_scalar(0.1), _device_scalar(1e-14)),
    )

    def run_solver(x_init):
        val0, grad0 = val_and_grad_fn(x_init)
        norm0 = jnp.linalg.norm(grad0)

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (state["norm"] > tol_value)
                & (~state["stalled"])
            )

        def body_fun(state):
            dx, linear_residual, _ = _gmres_solve_newton_system(
                hvp_fn,
                state["x"],
                state["grad"],
                stab=stab,
                tol=linear_tol,
            )
            linear_residual_norm = jnp.linalg.norm(linear_residual)
            dense_threshold = jnp.maximum(1e-10, 1e-3 * state["norm"])

            def use_dense_fallback(_):
                H_solve = _stabilize_dense_hessian(
                    _materialize_dense_hessian(hvp_fn, state["x"]),
                    stab,
                )
                dx_dense = jnp.linalg.solve(H_solve, state["grad"])
                residual_dense = state["grad"] - H_solve @ dx_dense
                return dx_dense, residual_dense, jnp.linalg.norm(residual_dense)

            def keep_gmres_step(_):
                return dx, linear_residual, linear_residual_norm

            dx, linear_residual, linear_residual_norm = lax.cond(
                (~jnp.all(jnp.isfinite(dx))) | (linear_residual_norm > dense_threshold),
                use_dense_fallback,
                keep_gmres_step,
                operand=None,
            )

            def add_correction(current_dx):
                correction, _, _ = _gmres_solve_newton_system(
                    hvp_fn,
                    state["x"],
                    linear_residual,
                    stab=stab,
                    tol=linear_tol,
                )
                return lax.cond(
                    jnp.all(jnp.isfinite(correction)),
                    lambda corr: current_dx + corr,
                    lambda _corr: current_dx,
                    correction,
                )

            dx = lax.cond(
                linear_residual_norm > linear_tol,
                add_correction,
                lambda current_dx: current_dx,
                dx,
            )

            x_next = state["x"] - dx
            val_next, grad_next = val_and_grad_fn(x_next)
            accepted, candidate_norm = _newton_candidate_status(
                state["norm"], x_next, grad_next
            )
            next_nit = state["nit"] + 1
            if progress_callback is not None:
                lax.cond(
                    accepted,
                    lambda _: jax.debug.callback(
                        progress_callback,
                        next_nit,
                        val_next,
                        candidate_norm,
                    ),
                    lambda _: None,
                    operand=None,
                )
            return {
                "x": lax.select(accepted, x_next, state["x"]),
                "val": lax.select(accepted, val_next, state["val"]),
                "grad": lax.select(accepted, grad_next, state["grad"]),
                "norm": lax.select(accepted, candidate_norm, state["norm"]),
                "nit": lax.select(accepted, next_nit, state["nit"]),
                "stalled": ~accepted,
            }

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "val": val0,
                "grad": grad0,
                "norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
                "stalled": jnp.asarray(False),
            },
        )

        val_final, grad_final = val_and_grad_fn(state["x"])
        norm_final = jnp.linalg.norm(grad_final)
        H = _stabilize_dense_hessian(
            _materialize_dense_hessian(hvp_fn, state["x"]),
            stab,
        )

        return {
            "x": state["x"],
            "fun": val_final,
            "grad": grad_final,
            "hessian": H,
            "nit": state["nit"],
            "success": norm_final <= tol_value,
        }

    return jax.jit(run_solver)(x0)


def newton_exact(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
):
    """Newton solver for the exact Boozer residual system ``r(x) = 0``."""
    jac_fn = jax.jit(jax.jacfwd(residual_fn))
    res_fn = jax.jit(residual_fn)

    x = x0
    r = res_fn(x)
    J = jac_fn(x)
    norm = jnp.linalg.norm(r)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        dx = jnp.linalg.solve(J, r)
        dx = dx + jnp.linalg.solve(J, r - J @ dx)
        x = x - dx
        r = res_fn(x)
        J = jac_fn(x)
        norm = jnp.linalg.norm(r)
        nit += 1

    return {
        "x": x,
        "residual": r,
        "jacobian": J,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }


def newton_exact_traceable(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
):
    """Trace-safe Newton solver for the exact Boozer residual system."""
    jac_fn = jax.jacfwd(residual_fn)

    def run_solver(x_init):
        r0 = residual_fn(x_init)
        J0 = jac_fn(x_init)
        norm0 = jnp.linalg.norm(r0)

        def cond_fun(state):
            return (state["nit"] < maxiter) & (state["norm"] > tol)

        def body_fun(state):
            dx = jnp.linalg.solve(state["jacobian"], state["residual"])
            correction = jnp.linalg.solve(
                state["jacobian"],
                state["residual"] - state["jacobian"] @ dx,
            )
            x_next = state["x"] - (dx + correction)
            residual_next = residual_fn(x_next)
            jacobian_next = jac_fn(x_next)
            return {
                "x": x_next,
                "residual": residual_next,
                "jacobian": jacobian_next,
                "norm": jnp.linalg.norm(residual_next),
                "nit": state["nit"] + 1,
            }

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "residual": r0,
                "jacobian": J0,
                "norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
            },
        )

        return {
            "x": state["x"],
            "residual": state["residual"],
            "jacobian": state["jacobian"],
            "nit": state["nit"],
            "success": state["norm"] <= tol,
        }

    return jax.jit(run_solver)(x0)


# ---------------------------------------------------------------------------
# Dispatcher — shared hub for all optimizer methods
# ---------------------------------------------------------------------------


def _require_private_package(method):
    """Raise ImportError when absent; populate module globals on first call."""
    pkg = _load_private_pkg()
    if pkg is False:
        raise ImportError(
            f"Method {method!r} requires the private optimizer package "
            f"(simsopt.geo.optimizer_jax_private). "
            f"Install with: pip install -e ."
        )
    for name in _PRIVATE_LAZY_NAMES:
        if name not in globals():
            globals()[name] = getattr(pkg, name)


def jax_least_squares(
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
    """Least-squares optimizer adapter for residual-vector objectives.

    Contract by method family:

    - ``lm``:
      host-driven Levenberg-Marquardt reference lane.
    - ``lm-ondevice``:
      trace-safe Levenberg-Marquardt target lane.

    ``x0`` may be either the historical flat 1-D vector or a structured pytree.
    Structured states are flattened once inside this adapter, then callbacks and
    ``OptimizeResult.x`` / ``OptimizeResult.jac`` are restored to the original
    pytree structure on return. ``result.jac`` is the final least-squares
    gradient vector, not the residual Jacobian matrix; the dense residual
    Jacobian is exposed separately as ``result.residual_jacobian``.
    """
    if method not in _SUPPORTED_LEAST_SQUARES_METHODS:
        raise ValueError(
            "Unknown least-squares method "
            f"{method!r}. Supported: {sorted(_SUPPORTED_LEAST_SQUARES_METHODS)}."
        )

    residual_fn, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
        residual_fn,
        x0,
        value_and_grad=False,
        callback=callback,
    )

    def finalize(result):
        return _finalize_optimizer_result(result, pytree_adapter)

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    if method == "lm":
        _raise_if_strict_optimizer_fallback(
            method=method,
            detail=_STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
        )
        result = levenberg_marquardt(
            residual_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
    else:
        require_target_backend_x64("ondevice")
        result = levenberg_marquardt_traceable(
            residual_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )

    optimize_result = OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        residual=result["residual"],
        residual_jacobian=result["residual_jacobian"],
        hessian=result["hessian"],
        damping=result["damping"],
        nit=int(result["nit"]),
        nfev=int(result["nit"]) + 1,
        njev=int(result["nit"]) + 1,
        status=int(result["status"]),
        success=bool(result["success"]),
        message=_least_squares_result_message(
            result["status"],
            result["success"],
        ),
    )
    return finalize(optimize_result)


def jax_minimize(
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
    """Optimizer adapter for Boozer LS minimization.

    Contract by method family:

    - ``bfgs`` / ``lbfgs``:
      trusted reference/oracle path using host-side SciPy loops.
    - ``bfgs-hybrid``:
      transitional private path for staged migration away from SciPy.
    - ``bfgs-ondevice`` / ``lbfgs-ondevice``:
      private target path for the eventual full-GPU optimizer lane.

    If ``value_and_grad=True``, ``fun`` must return ``(value, grad)`` directly.
    That explicit value/gradient contract is supported on the trusted SciPy
    reference methods and on the ``lbfgs-ondevice`` target method used by the
    single-stage outer loop. The ``lbfgs-ondevice`` explicit path expects a
    JAX-traceable callable and routes through the private on-device
    implementation directly.

    ``x0`` may be either the historical flat 1-D vector or a structured pytree.
    Structured states are flattened once inside this adapter, then callbacks and
    ``OptimizeResult.x`` / ``OptimizeResult.jac`` are restored to the original
    pytree structure on return.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )

    fun, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
        fun,
        x0,
        value_and_grad=value_and_grad,
        callback=callback,
    )

    def finalize(result):
        return _finalize_optimizer_result(result, pytree_adapter)

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    if method in _REFERENCE_METHODS:
        _raise_if_strict_optimizer_fallback(
            method=method,
            detail=_STRICT_REFERENCE_OPTIMIZER_DETAIL,
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

    # All remaining methods require the private optimizer package.
    _require_private_package(method)

    if value_and_grad:
        if method != "lbfgs-ondevice":
            raise RuntimeError(
                "Explicit value-and-gradient objectives are only supported on the "
                "trusted SciPy reference methods and lbfgs-ondevice today."
            )
        state = _minimize_lbfgs_private_value_and_grad(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 200)),
            ftol=float(options.get("ftol", 0.0)),
            maxfun=options.get("maxfun"),
            maxgrad=options.get("maxgrad"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
        return finalize(_private_lbfgs_result_to_optimize_result(state))

    if method == "bfgs-ondevice":
        state = _minimize_bfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            line_search_maxiter=int(options.get("line_search_maxiter", 10)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
        return finalize(_private_bfgs_result_to_optimize_result(state))

    if method == "lbfgs-ondevice":
        state = _minimize_lbfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 200)),
            ftol=float(options.get("ftol", 0.0)),
            maxfun=options.get("maxfun"),
            maxgrad=options.get("maxgrad"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
        )
        return finalize(_private_lbfgs_result_to_optimize_result(state))

    # --- bfgs-hybrid: SciPy prefix → on-device continuation ---
    _raise_if_strict_optimizer_fallback(
        method=method,
        detail=_STRICT_HYBRID_OPTIMIZER_DETAIL,
    )
    total_maxiter = int(maxiter)
    prefix_cap = int(options.get("hybrid_scipy_maxiter", min(total_maxiter // 2, 100)))
    prefix_cap = max(0, min(prefix_cap, total_maxiter - 1))
    prefix_result = _scipy_minimize(
        fun,
        x0,
        method="bfgs",
        tol=tol,
        maxiter=prefix_cap,
        options=options,
    )
    if prefix_result.success:
        return finalize(prefix_result)
    if not _scipy_result_is_continuable(prefix_result):
        prefix_result.success = False
        prefix_result.message = (
            "SciPy prefix produced a non-finite state; on-device continuation skipped."
        )
        return finalize(prefix_result)

    remaining_maxiter = max(0, total_maxiter - int(prefix_result.nit))
    if remaining_maxiter == 0:
        prefix_result.success = False
        return finalize(prefix_result)

    continuation_state = _make_bfgs_continuation_state(
        prefix_result,
        gtol=tol,
        norm=np.inf,
    )
    final_state = _minimize_bfgs_private(
        fun,
        prefix_result.x,
        maxiter=remaining_maxiter,
        gtol=tol,
        line_search_maxiter=int(options.get("line_search_maxiter", 10)),
        initial_state=continuation_state,
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
    )
    result = _private_bfgs_result_to_optimize_result(
        final_state,
        total_nit=int(prefix_result.nit) + int(final_state.k),
    )
    result.hess_inv = jnp.asarray(final_state.H_k)
    return finalize(result)
