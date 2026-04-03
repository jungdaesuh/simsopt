"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.

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

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import minimize as scipy_minimize

from ..backend import raise_if_strict_jax_fallback

__all__ = [
    "ContinuousOptimizerContract",
    "PRIVATE_OPTIMIZER_JAX_VERSION",
    "VALID_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "jax_minimize",
    "newton_polish",
    "newton_polish_traceable",
    "newton_exact",
    "newton_exact_traceable",
    "require_target_backend_x64",
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
_SUPPORTED_METHODS = {
    "bfgs",
    "lbfgs",
    "bfgs-hybrid",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_HYBRID_OPTIMIZER_DETAIL = "the transitional SciPy-prefix hybrid optimizer lane"


@dataclass(frozen=True)
class ContinuousOptimizerContract:
    method: str
    use_scalar_objective: bool


def _raise_if_strict_optimizer_fallback(*, method: str, detail: str) -> None:
    raise_if_strict_jax_fallback(
        component="optimizer_jax.jax_minimize",
        detail=f"{detail} for method={method!r}",
    )


def _x64_enabled():
    return bool(jnp.zeros(1).dtype == jnp.float64)


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
        "_minimize_lbfgs_explicit_value_and_grad",
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
    val0, grad0 = val_and_grad_fn(x0)
    norm0 = jnp.linalg.norm(grad0)
    linear_tol = jnp.minimum(1e-10, jnp.maximum(jnp.asarray(tol) * 0.1, 1e-14))

    def cond_fun(state):
        return (state["nit"] < maxiter) & (state["norm"] > tol) & (~state["stalled"])

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
            "x": x0,
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
        "success": norm_final <= tol,
    }


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

    r0 = residual_fn(x0)
    J0 = jac_fn(x0)
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
            "x": x0,
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
    JAX-traceable callable; host-loop NumPy fallbacks remain private helpers,
    not the production target lane.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )

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
        return scipy_adapter(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
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
        return _private_lbfgs_result_to_optimize_result(state)

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
        return _private_bfgs_result_to_optimize_result(state)

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
        return _private_lbfgs_result_to_optimize_result(state)

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
        return prefix_result
    if not _scipy_result_is_continuable(prefix_result):
        prefix_result.success = False
        prefix_result.message = (
            "SciPy prefix produced a non-finite state; on-device continuation skipped."
        )
        return prefix_result

    remaining_maxiter = max(0, total_maxiter - int(prefix_result.nit))
    if remaining_maxiter == 0:
        prefix_result.success = False
        return prefix_result

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
    return result
