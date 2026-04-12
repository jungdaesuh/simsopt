"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.
  - ``method="adam"``: host-driven Adam for noisy/stochastic scalar objectives.

Least-squares methods:
  - ``method="lm"``: host-driven Levenberg-Marquardt for residual-vector
    objectives on the reference lane.
  - ``method="lm-ondevice"``: trace-safe Levenberg-Marquardt for
    residual-vector objectives on the target lane.

Target private methods (minimum supported JAX floor 0.9.2):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: JAX on-device L-BFGS.

Target public stochastic method:
  - ``method="adam-ondevice"``: trace-safe Adam for noisy/stochastic scalar
    objectives on the target lane.

The private methods live in ``optimizer_jax_private/`` and intentionally mirror
the JAX 0.9.2 optimizer semantics so the line-search and iteration behavior
stay stable across this project. High-level JAX backend flows route through the
target lane only; the host SciPy adapter lives in the separate
``optimizer_jax_reference`` module. The reference source is the upstream
``jax-v0.9.2`` tag (``a659757d768587a81d095a9fab5f0c36f8beb218``).

This module contains zero ``jax._src`` imports. The private package now does as
well; both paths use public JAX APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from threading import Lock
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import lax
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import OptimizeResult

from ..backend import (
    get_backend_config,
    raise_if_strict_jax_fallback,
)
from .._core.jax_host_boundary import host_bool as _host_bool
from .._core.jax_host_boundary import host_scalar as _host_scalar
from ..jax_core._math_utils import _explicit_device_array

__all__ = [
    "PRIVATE_OPTIMIZER_JAX_VERSION",
    "ReferenceOptimizerContract",
    "TargetOptimizerContract",
    "adam_optimize",
    "adam_optimize_traceable",
    "private_optimizer_runtime_is_supported",
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
    "reference_least_squares",
    "reference_minimize",
    "require_target_backend_x64",
    "resolve_least_squares_optimizer_method",
    "resolve_reference_least_squares_optimizer_method",
    "resolve_reference_optimizer_contract",
    "resolve_reference_optimizer_method",
    "resolve_target_least_squares_optimizer_method",
    "resolve_target_optimizer_contract",
    "resolve_target_optimizer_method",
    "resolve_optimizer_backend_method",
    "resolve_reference_outer_loop_optimizer_contract",
    "resolve_target_outer_loop_optimizer_contract",
    "target_least_squares",
    "target_minimize",
]


PRIVATE_OPTIMIZER_JAX_VERSION = "0.9.2"
VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})
OPTIMIZER_BACKEND_ROLE = {
    "scipy": "reference",
    "ondevice": "target",
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = frozenset({"ondevice"})
VALID_LEAST_SQUARES_ALGORITHMS = frozenset({"quasi-newton", "lm"})
_SUPPORTED_METHODS = {
    "adam",
    "adam-ondevice",
    "bfgs",
    "lbfgs",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
_SUPPORTED_LEAST_SQUARES_METHODS = frozenset({"lm", "lm-ondevice"})
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_REFERENCE_JAX_METHODS = frozenset({"adam"})
_TARGET_PRIVATE_METHODS = frozenset({"bfgs-ondevice", "lbfgs-ondevice"})
_TARGET_PUBLIC_METHODS = frozenset({"adam-ondevice"})
_TARGET_METHODS = _TARGET_PRIVATE_METHODS | _TARGET_PUBLIC_METHODS
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL = "the host-side JAX reference optimizer lane"
_STRICT_REFERENCE_LEAST_SQUARES_DETAIL = (
    "the host-side reference least-squares optimizer lane"
)
_SCALAR_VALUE_AND_GRAD_CACHE_LOCK = Lock()
_CACHEABLE_VALUE_AND_GRAD_ATTR = "_simsopt_cache_jit_value_and_grad"
_CACHED_VALUE_AND_GRAD_ATTR = "_simsopt_cached_jit_value_and_grad"


def _version_key(raw_version: str) -> tuple[int, ...]:
    base = raw_version.split("+", 1)[0]
    parts = re.findall(r"\d+", base)
    return tuple(int(part) for part in parts)


def private_optimizer_runtime_is_supported(version: str) -> bool:
    """Return True when the runtime meets the minimum supported JAX floor."""
    return _version_key(version) >= _version_key(PRIVATE_OPTIMIZER_JAX_VERSION)


@dataclass(frozen=True)
class ReferenceOptimizerContract:
    method: str


@dataclass(frozen=True)
class TargetOptimizerContract:
    method: str
    use_least_squares_objective: bool = False


@dataclass(frozen=True)
class _OptimizerPytreeAdapter:
    flat_dtype: np.dtype
    unravel: Callable[[jax.Array], object]
    tree_def: object

    def _hostify_flat(self, flat_x, *, dtype=None):
        return _hostify_optimizer_tree(
            self.unravel(_optimizer_flat_vector(flat_x, dtype=dtype))
        )

    def wrap_fun(self, fun, *, value_and_grad: bool):
        if not value_and_grad:

            def wrapped(flat_x):
                return fun(self.unravel(_optimizer_flat_vector(flat_x)))

            return wrapped

        def wrapped(flat_x):
            flat_x = _optimizer_flat_vector(flat_x)
            value, grad_tree = fun(self.unravel(flat_x))
            _, grad_tree_def = jax.tree_util.tree_flatten(grad_tree)
            if grad_tree_def != self.tree_def:
                raise ValueError(
                    "Explicit value-and-gradient objectives must return a gradient "
                    "with the same pytree structure as x0."
                )
            flat_grad, _ = ravel_pytree(grad_tree)
            flat_grad = _optimizer_flat_vector(flat_grad, dtype=flat_x.dtype)
            if flat_grad.shape != flat_x.shape:
                raise ValueError(
                    "Explicit value-and-gradient objectives must return a gradient "
                    f"matching the flattened x0 shape {flat_x.shape}, got {flat_grad.shape}."
                )
            return _optimizer_scalar(value, dtype=flat_x.dtype), flat_grad

        return wrapped

    def wrap_callback(self, callback):
        if callback is None:
            return None

        def wrapped(flat_x):
            callback(self._hostify_flat(flat_x))

        return wrapped

    def finalize_result(self, result):
        if hasattr(result, "x"):
            result.x = self._hostify_flat(result.x, dtype=self.flat_dtype)
        if hasattr(result, "jac"):
            result.jac = self._hostify_flat(result.jac, dtype=self.flat_dtype)
        return result


def _raise_if_strict_optimizer_fallback(
    *,
    component: str,
    method: str,
    detail: str,
) -> None:
    raise_if_strict_jax_fallback(
        component=component,
        detail=f"{detail} for method={method!r}",
    )


def _raise_if_target_lane_required(
    *,
    component: str,
    method: str,
    detail: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use {detail} for method={method!r} while simsopt "
        f"backend mode {backend_config.mode!r} requires an ondevice optimizer "
        "method. Select an ondevice optimizer method or switch to the "
        "native_cpu reference backend."
    )


def _require_native_cpu_reference_backend_for_scipy_adapter(
    *,
    component: str,
    method: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use the host SciPy adapter for method={method!r} "
        f"while simsopt backend mode {backend_config.mode!r} requires an "
        "ondevice optimizer method. Select an ondevice optimizer method or "
        "switch to the native_cpu reference backend."
    )


def _x64_enabled():
    return bool(jax.config.jax_enable_x64)


def _device_scalar(value, *, dtype=jnp.float64):
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _optimizer_flat_vector(value, *, dtype=None) -> jax.Array:
    if isinstance(value, jax.Array):
        if dtype is None or value.dtype == dtype:
            return value
        return jnp.asarray(value, dtype=dtype)
    if hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return _explicit_device_array(value, dtype=dtype)


def _optimizer_scalar(value, *, dtype) -> jax.Array:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    return _explicit_device_array(value, dtype=dtype)


def _optimizer_dtype(value):
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return dtype
    return np.asarray(value).dtype


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
    flat_dtype = np.dtype(_optimizer_dtype(flat_x0))
    return _OptimizerPytreeAdapter(
        flat_dtype=flat_dtype,
        unravel=unravel,
        tree_def=tree_def,
    )


def _mark_cacheable_jit_value_and_grad(fun):
    try:
        setattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    except (AttributeError, TypeError):
        pass
    return fun


def _cached_jit_value_and_grad(fun):
    if not getattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, False):
        return jax.jit(jax.value_and_grad(fun))
    cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
    if cached is not None:
        return cached
    compiled = jax.jit(jax.value_and_grad(fun))
    try:
        with _SCALAR_VALUE_AND_GRAD_CACHE_LOCK:
            cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
            if cached is None:
                setattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, compiled)
                return compiled
            return cached
    except (AttributeError, TypeError):
        return compiled


def _prepare_optimizer_callable_inputs(fun, x0, *, value_and_grad, callback):
    adapter = _prepare_optimizer_pytree_adapter(x0)
    if adapter is None:
        return fun, x0, callback, None
    flat_x0, _ = ravel_pytree(x0)
    return (
        adapter.wrap_fun(fun, value_and_grad=value_and_grad),
        _optimizer_flat_vector(flat_x0, dtype=adapter.flat_dtype),
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
        "_minimize_bfgs_private",
        "_minimize_lbfgs_private",
        "_minimize_lbfgs_private_value_and_grad",
        "_private_bfgs_result_to_optimize_result",
        "_private_lbfgs_result_to_optimize_result",
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
        raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
    if optimizer_backend == "scipy":
        return resolve_reference_optimizer_method(limited_memory=limited_memory)
    return resolve_target_optimizer_method(limited_memory=limited_memory)


def resolve_reference_optimizer_method(*, limited_memory):
    """Resolve the CPU/reference scalar optimizer method."""
    return "lbfgs" if limited_memory else "bfgs"


def resolve_target_optimizer_method(*, limited_memory):
    """Resolve the JAX target scalar optimizer method."""
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
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")
    if least_squares_algorithm == "quasi-newton":
        return resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=limited_memory,
        )
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
    if optimizer_backend == "scipy":
        return resolve_reference_least_squares_optimizer_method(
            limited_memory=limited_memory,
            least_squares_algorithm=least_squares_algorithm,
        )
    return resolve_target_least_squares_optimizer_method(
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )


def resolve_reference_least_squares_optimizer_method(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the CPU/reference least-squares optimizer method."""
    if least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS:
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")
    if least_squares_algorithm == "quasi-newton":
        return resolve_reference_optimizer_method(limited_memory=limited_memory)
    if limited_memory:
        raise ValueError(
            "least_squares_algorithm='lm' is incompatible with limited_memory=True."
        )
    return "lm"


def resolve_target_least_squares_optimizer_method(
    *,
    limited_memory,
    least_squares_algorithm,
):
    """Resolve the JAX target least-squares optimizer method."""
    if least_squares_algorithm not in VALID_LEAST_SQUARES_ALGORITHMS:
        allowed = ", ".join(sorted(VALID_LEAST_SQUARES_ALGORITHMS))
        raise ValueError(f"least_squares_algorithm must be one of: {allowed}.")
    if least_squares_algorithm == "quasi-newton":
        return resolve_target_optimizer_method(limited_memory=limited_memory)
    if limited_memory:
        raise ValueError(
            "least_squares_algorithm='lm' is incompatible with limited_memory=True."
        )
    return "lm-ondevice"


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


def resolve_reference_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    limited_memory,
    component_label,
):
    """Resolve the explicit CPU/reference optimizer contract."""
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
    if field_backend == "jax":
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice'. The SciPy/reference optimizer lane "
            "is CPU/reference-only."
        )
    if field_backend != "jax" and optimizer_backend != "scipy":
        raise ValueError(
            f"{component_label} CPU/reference lane only supports "
            "optimizer_backend='scipy'."
        )
    return ReferenceOptimizerContract(
        method=resolve_reference_optimizer_method(
            limited_memory=limited_memory,
        ),
    )


def resolve_target_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    limited_memory,
    component_label,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the explicit JAX target optimizer contract."""
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend must be one of: scipy, ondevice.")
    if field_backend != "jax" or optimizer_backend != "ondevice":
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice'. The SciPy/reference optimizer lane "
            "is CPU/reference-only."
        )
    require_target_backend_x64(optimizer_backend)
    method = resolve_target_least_squares_optimizer_method(
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )
    return TargetOptimizerContract(
        method=method,
        use_least_squares_objective=(method == "lm-ondevice"),
    )


def resolve_reference_outer_loop_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    component_label,
):
    """Resolve the CPU/reference outer-loop contract."""
    return resolve_reference_optimizer_contract(
        field_backend,
        optimizer_backend,
        limited_memory=True,
        component_label=component_label,
    )


def resolve_target_outer_loop_optimizer_contract(
    field_backend,
    optimizer_backend,
    *,
    component_label,
    least_squares_algorithm="quasi-newton",
):
    """Resolve the JAX target outer-loop contract."""
    limited_memory = least_squares_algorithm != "lm"
    return resolve_target_optimizer_contract(
        field_backend,
        optimizer_backend,
        limited_memory=limited_memory,
        component_label=component_label,
        least_squares_algorithm=least_squares_algorithm,
    )


# ---------------------------------------------------------------------------
# Reference-lane module loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_reference_optimizer_module():
    from . import optimizer_jax_reference

    return optimizer_jax_reference


def _least_squares_cost(residual):
    residual = jnp.ravel(jnp.asarray(residual))
    return _device_scalar(0.5, dtype=residual.dtype) * jnp.vdot(residual, residual).real


def _least_squares_linearization_from_jacobian(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    return gradient, hessian


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(jnp.asarray(leaf)),
        tree,
    )


def _tree_scalar_mul(tree, scalar):
    scalar = jnp.asarray(scalar)
    return jax.tree_util.tree_map(lambda leaf: scalar * jnp.asarray(leaf), tree)


def _tree_add(lhs, rhs):
    return jax.tree_util.tree_map(
        lambda lhs_leaf, rhs_leaf: jnp.asarray(lhs_leaf) + jnp.asarray(rhs_leaf),
        lhs,
        rhs,
    )


def _tree_sub(lhs, rhs):
    return jax.tree_util.tree_map(
        lambda lhs_leaf, rhs_leaf: jnp.asarray(lhs_leaf) - jnp.asarray(rhs_leaf),
        lhs,
        rhs,
    )


def _tree_square(tree):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.square(jnp.asarray(leaf)),
        tree,
    )


def _tree_bias_correction(tree, correction):
    correction = jnp.asarray(correction)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf) / correction,
        tree,
    )


def _tree_adam_step(mean, variance, *, step_size, eps):
    step_size = jnp.asarray(step_size)
    eps = jnp.asarray(eps)
    return jax.tree_util.tree_map(
        lambda mean_leaf, variance_leaf: (
            step_size
            * jnp.asarray(mean_leaf)
            / (jnp.sqrt(jnp.asarray(variance_leaf)) + eps)
        ),
        mean,
        variance,
    )


def _require_tree_first_leaf(tree, *, detail):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(detail)
    return jnp.asarray(leaves[0])


def _tree_vdot_real(lhs, rhs):
    lhs_leaves, lhs_tree = jax.tree_util.tree_flatten(lhs)
    rhs_leaves, rhs_tree = jax.tree_util.tree_flatten(rhs)
    if lhs_tree != rhs_tree:
        raise ValueError("Tree dot products require matching pytree structures.")
    if not lhs_leaves:
        return _device_scalar(0.0)
    dtype = jnp.result_type(
        *[jnp.asarray(leaf).dtype for leaf in lhs_leaves + rhs_leaves]
    )
    total = jnp.asarray(0.0, dtype=dtype)
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        total = total + jnp.vdot(
            jnp.ravel(jnp.asarray(lhs_leaf)),
            jnp.ravel(jnp.asarray(rhs_leaf)),
        ).real.astype(dtype)
    return total


def _tree_inf_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return _device_scalar(0.0)
    dtype = jnp.result_type(*[jnp.asarray(leaf).dtype for leaf in leaves])
    max_value = jnp.asarray(0.0, dtype=dtype)
    for leaf in leaves:
        leaf = jnp.ravel(jnp.asarray(leaf))
        leaf_norm = jnp.asarray(0.0, dtype=dtype)
        if leaf.size:
            leaf_norm = jnp.max(jnp.abs(leaf)).astype(dtype)
        max_value = jnp.maximum(max_value, leaf_norm)
    return max_value


def _tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    finite = jnp.asarray(True)
    for leaf in leaves:
        finite = finite & jnp.all(jnp.isfinite(jnp.asarray(leaf)))
    return finite


def _tree_select(pred, candidate, current):
    return jax.tree_util.tree_map(
        lambda cand, curr: lax.select(pred, jnp.asarray(cand), jnp.asarray(curr)),
        candidate,
        current,
    )


def _flattened_residual_output(residual_fn):
    def wrapped(x):
        return jnp.ravel(jnp.asarray(residual_fn(x)))

    return wrapped


def _normalize_solver_args(args):
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    return (args,)


def _wrap_value_and_grad_fun(fun, x0, *, host_inputs):
    expected_tree = jax.tree_util.tree_structure(x0)

    def wrapped(x):
        call_x = _hostify_optimizer_tree(x) if host_inputs else x
        value, grad = fun(call_x)
        if jax.tree_util.tree_structure(grad) != expected_tree:
            raise ValueError(
                "Explicit value-and-gradient objectives must return a gradient "
                "with the same pytree structure as x0."
            )
        return jnp.asarray(value), jax.tree_util.tree_map(jnp.asarray, grad)

    return wrapped


def _prepare_adam_eval_fn(fun, x0, *, value_and_grad, host_inputs):
    if value_and_grad:
        return _wrap_value_and_grad_fun(fun, x0, host_inputs=host_inputs)
    return _cached_jit_value_and_grad(fun)


def _adam_defaults(dtype):
    return {
        "step_size": _device_scalar(1.0e-2, dtype=dtype),
        "beta1": _device_scalar(0.9, dtype=dtype),
        "beta2": _device_scalar(0.999, dtype=dtype),
        "eps": _device_scalar(1.0e-8, dtype=dtype),
    }


def _adam_hyperparameters(options, *, dtype):
    defaults = _adam_defaults(dtype)
    options = options or {}
    return {
        "step_size": _device_scalar(
            options.get("step_size", defaults["step_size"]), dtype=dtype
        ),
        "beta1": _device_scalar(options.get("beta1", defaults["beta1"]), dtype=dtype),
        "beta2": _device_scalar(options.get("beta2", defaults["beta2"]), dtype=dtype),
        "eps": _device_scalar(options.get("eps", defaults["eps"]), dtype=dtype),
    }


def _adam_result_message(status, success):
    if _host_bool(success):
        return "converged"
    if int(_host_scalar(status, dtype=np.int64)) == 2:
        return "non-finite objective, gradient, or step encountered"
    return "maximum iterations reached"


def _adam_result_to_optimize_result(result):
    nit = int(_host_scalar(result["nit"], dtype=np.int64))
    status = int(_host_scalar(result["status"], dtype=np.int64))
    success = _host_bool(result["success"])
    return OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        nit=nit,
        nfev=nit + 1,
        njev=nit + 1,
        status=status,
        success=success,
        mean=result["mean"],
        variance=result["variance"],
        message=_adam_result_message(status, success),
    )


def _adam_iteration(eval_fn, state, *, hyperparameters, tol):
    step_number = state["nit"] + 1
    beta1 = hyperparameters["beta1"]
    beta2 = hyperparameters["beta2"]
    one_minus_beta1 = jnp.asarray(1.0, dtype=beta1.dtype) - beta1
    one_minus_beta2 = jnp.asarray(1.0, dtype=beta2.dtype) - beta2
    mean = _tree_add(
        _tree_scalar_mul(state["mean"], beta1),
        _tree_scalar_mul(state["grad"], one_minus_beta1),
    )
    variance = _tree_add(
        _tree_scalar_mul(state["variance"], beta2),
        _tree_scalar_mul(_tree_square(state["grad"]), one_minus_beta2),
    )
    step_exponent = jnp.asarray(step_number, dtype=beta1.dtype)
    mean_hat = _tree_bias_correction(mean, 1.0 - jnp.power(beta1, step_exponent))
    variance_hat = _tree_bias_correction(
        variance,
        1.0 - jnp.power(beta2, step_exponent),
    )
    step = _tree_adam_step(
        mean_hat,
        variance_hat,
        step_size=hyperparameters["step_size"],
        eps=hyperparameters["eps"],
    )
    x_candidate = _tree_sub(state["x"], step)
    fun_candidate, grad_candidate = eval_fn(x_candidate)
    grad_norm_inf = _tree_inf_norm(grad_candidate)
    finite_candidate = (
        _tree_all_finite(x_candidate)
        & jnp.isfinite(fun_candidate)
        & _tree_all_finite(grad_candidate)
        & _tree_all_finite(step)
    )
    return {
        "x": _tree_select(finite_candidate, x_candidate, state["x"]),
        "fun": lax.select(finite_candidate, fun_candidate, state["fun"]),
        "grad": _tree_select(finite_candidate, grad_candidate, state["grad"]),
        "grad_norm_inf": lax.select(
            finite_candidate,
            grad_norm_inf,
            state["grad_norm_inf"],
        ),
        "mean": _tree_select(finite_candidate, mean, state["mean"]),
        "variance": _tree_select(finite_candidate, variance, state["variance"]),
        "nit": step_number,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "success": finite_candidate & (grad_norm_inf <= tol),
    }


def adam_optimize(
    fun,
    x0,
    *,
    value_and_grad=False,
    maxiter=1500,
    tol=1e-10,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Host-driven Adam optimizer for noisy/stochastic scalar objectives."""
    x = jax.tree_util.tree_map(jnp.asarray, x0)
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Adam initial state must contain at least one leaf.",
    ).dtype
    eval_fn = _prepare_adam_eval_fn(
        fun, x, value_and_grad=value_and_grad, host_inputs=True
    )
    hyperparameters = _adam_hyperparameters(options, dtype=x_dtype)
    fun_value, grad = eval_fn(x)
    grad_norm_inf = _tree_inf_norm(grad)
    mean = _tree_zeros_like(x)
    variance = _tree_zeros_like(x)
    nit = 0
    status = 1
    success = bool(grad_norm_inf <= tol)

    while nit < maxiter and not success:
        state = _adam_iteration(
            eval_fn,
            {
                "x": x,
                "fun": fun_value,
                "grad": grad,
                "grad_norm_inf": grad_norm_inf,
                "mean": mean,
                "variance": variance,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
            },
            hyperparameters=hyperparameters,
            tol=_device_scalar(tol, dtype=x_dtype),
        )
        nit = int(state["nit"])
        status = int(state["status"])
        x = state["x"]
        fun_value = state["fun"]
        grad = state["grad"]
        grad_norm_inf = state["grad_norm_inf"]
        mean = state["mean"]
        variance = state["variance"]
        if callback is not None:
            callback(_hostify_optimizer_tree(x))
        if progress_callback is not None:
            progress_callback(nit, float(fun_value), float(grad_norm_inf))
        success = bool(state["success"])
        if status == 2:
            break

    return {
        "x": x,
        "fun": fun_value,
        "grad": grad,
        "mean": mean,
        "variance": variance,
        "nit": nit,
        "status": status,
        "success": success,
    }


def adam_optimize_traceable(
    fun,
    x0,
    *,
    value_and_grad=False,
    maxiter=1500,
    tol=1e-10,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Trace-safe Adam optimizer for noisy/stochastic scalar objectives."""
    x = jax.tree_util.tree_map(jnp.asarray, x0)
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Adam initial state must contain at least one leaf.",
    ).dtype
    eval_fn = _prepare_adam_eval_fn(
        fun, x, value_and_grad=value_and_grad, host_inputs=False
    )
    hyperparameters = _adam_hyperparameters(options, dtype=x_dtype)
    tol_value = _device_scalar(tol, dtype=x_dtype)

    def run_solver(x_init):
        fun0, grad0 = eval_fn(x_init)
        state0 = {
            "x": x_init,
            "fun": fun0,
            "grad": grad0,
            "grad_norm_inf": _tree_inf_norm(grad0),
            "mean": _tree_zeros_like(x_init),
            "variance": _tree_zeros_like(x_init),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(1, dtype=jnp.int32),
            "success": _tree_inf_norm(grad0) <= tol_value,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter) & (~state["success"]) & (state["status"] != 2)
            )

        def body_fun(state):
            next_state = _adam_iteration(
                eval_fn,
                state,
                hyperparameters=hyperparameters,
                tol=tol_value,
            )
            if callback is not None:
                jax.debug.callback(
                    lambda current_x: callback(_hostify_optimizer_tree(current_x)),
                    next_state["x"],
                )
            if progress_callback is not None:
                jax.debug.callback(
                    progress_callback,
                    next_state["nit"],
                    next_state["fun"],
                    next_state["grad_norm_inf"],
                )
            return next_state

        return lax.while_loop(cond_fun, body_fun, state0)

    return jax.jit(run_solver)(x)


def _least_squares_gradient_state(flat_residual_fn, x):
    residual, pullback = jax.vjp(flat_residual_fn, x)
    grad = pullback(residual)[0]
    cost = _least_squares_cost(residual)
    grad_norm_inf = _tree_inf_norm(grad)
    return residual, cost, grad, grad_norm_inf, pullback


@lru_cache(maxsize=128)
def _make_traceable_levenberg_marquardt_runner(
    residual_fn,
    maxiter,
    tol,
    callback,
    progress_callback,
):
    def run_solver(x_init, fn_args):
        def residual_eval(x):
            return jnp.ravel(jnp.asarray(residual_fn(x, *fn_args)))

        x_dtype = _require_tree_first_leaf(
            x_init,
            detail="Least-squares initial state must contain at least one leaf.",
        ).dtype
        tol_value = _device_scalar(tol, dtype=x_dtype)
        residual0, cost0, grad0, grad_norm_inf0, _ = _least_squares_gradient_state(
            residual_eval,
            x_init,
        )
        state0 = {
            "x": x_init,
            "residual": residual0,
            "cost": cost0,
            "grad": grad0,
            "grad_norm_inf": grad_norm_inf0,
            "damping": _lm_defaults(x_dtype)["initial_damping"],
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "accepted": jnp.asarray(False),
            "success": grad_norm_inf0 <= tol_value,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter) & (~state["success"]) & (state["status"] != 2)
            )

        def body_fun(state):
            next_state = _lm_iteration(
                residual_eval,
                state,
                tol=tol_value,
            )
            if callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        lambda x: callback(_hostify_optimizer_tree(x)),
                        next_state["x"],
                    ),
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
        residual_final, residual_jacobian, _flat_grad, hessian = (
            _materialize_dense_least_squares_linearization(
                residual_eval,
                state["x"],
            )
        )
        return {
            "x": state["x"],
            "residual": residual_final,
            "residual_jacobian": residual_jacobian,
            "fun": state["cost"],
            "grad": state["grad"],
            "hessian": hessian,
            "damping": state["damping"],
            "nit": state["nit"],
            "status": state["status"],
            "success": state["success"],
        }

    return jax.jit(run_solver)


def _least_squares_matvec(flat_residual_fn, x, pullback, tangent):
    jvp_residual = jax.jvp(flat_residual_fn, (x,), (tangent,))[1]
    return pullback(jvp_residual)[0]


def _gmres_solve_least_squares_system(
    flat_residual_fn,
    x,
    grad,
    pullback,
    *,
    damping,
    tol,
):
    grad_leaves = jax.tree_util.tree_leaves(grad)
    first_grad_leaf = _require_tree_first_leaf(
        grad,
        detail="Least-squares gradients must contain at least one leaf.",
    )
    dtype = first_grad_leaf.dtype
    n = sum(int(np.asarray(jnp.asarray(leaf).size)) for leaf in grad_leaves)
    restart = max(5, min(n, 50))
    maxiter = max(10, min(4 * n, 200))
    damping_value = jnp.asarray(damping, dtype=dtype)

    def matvec(v):
        jt_j_v = _least_squares_matvec(flat_residual_fn, x, pullback, v)
        return jax.tree_util.tree_map(
            lambda jt_j_leaf, v_leaf: jt_j_leaf + damping_value * v_leaf,
            jt_j_v,
            v,
        )

    step, _ = gmres(
        matvec,
        grad,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
    )
    residual = jax.tree_util.tree_map(
        lambda grad_leaf, matvec_leaf: grad_leaf - matvec_leaf,
        grad,
        matvec(step),
    )
    return step, residual, matvec


def _materialize_dense_least_squares_linearization(flat_residual_fn, x):
    flat_x, unravel = ravel_pytree(x)
    flat_x = jnp.asarray(flat_x)
    jvp_fn = _jacobian_vector_product_fn(lambda flat: flat_residual_fn(unravel(flat)))
    residual = flat_residual_fn(x)
    jacobian = _materialize_dense_jacobian(jvp_fn, flat_x)
    gradient, hessian = _least_squares_linearization_from_jacobian(
        residual,
        jacobian,
    )
    return residual, jacobian, gradient, hessian


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


def _lm_iteration(flat_residual_fn, state, *, tol):
    state_dtype = _require_tree_first_leaf(
        state["x"],
        detail="Least-squares state x must contain at least one leaf.",
    ).dtype
    defaults = _lm_defaults(state_dtype)
    damping = _clip_lm_damping(state["damping"], dtype=state_dtype)
    linear_tol = jnp.minimum(
        _device_scalar(1.0e-10, dtype=state["cost"].dtype),
        jnp.maximum(
            jnp.asarray(tol, dtype=state["cost"].dtype)
            * _device_scalar(0.1, dtype=state["cost"].dtype),
            _device_scalar(1.0e-14, dtype=state["cost"].dtype),
        ),
    )
    _, _, _, _, current_pullback = _least_squares_gradient_state(
        flat_residual_fn,
        state["x"],
    )
    step, linear_residual, _ = _gmres_solve_least_squares_system(
        flat_residual_fn,
        state["x"],
        state["grad"],
        current_pullback,
        damping=damping,
        tol=linear_tol,
    )
    x_candidate = jax.tree_util.tree_map(
        lambda x_leaf, step_leaf: x_leaf - step_leaf,
        state["x"],
        step,
    )
    residual_candidate, cost_candidate, grad_candidate, grad_norm_candidate, _ = (
        _least_squares_gradient_state(flat_residual_fn, x_candidate)
    )

    predicted_reduction = _device_scalar(
        0.5,
        dtype=state["cost"].dtype,
    ) * (
        jnp.asarray(damping, dtype=state["cost"].dtype) * _tree_vdot_real(step, step)
        + _tree_vdot_real(step, state["grad"])
    )
    actual_reduction = state["cost"] - cost_candidate
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        _tree_all_finite(x_candidate)
        & jnp.all(jnp.isfinite(residual_candidate))
        & jnp.isfinite(cost_candidate)
        & _tree_all_finite(grad_candidate)
        & _tree_all_finite(linear_residual)
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
        lambda _: _clip_lm_damping(damping_after_accept, dtype=state_dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["expand_factor"],
            dtype=state_dtype,
        ),
        operand=None,
    )

    return {
        "x": _tree_select(accepted, x_candidate, state["x"]),
        "residual": lax.select(accepted, residual_candidate, state["residual"]),
        "cost": lax.select(accepted, cost_candidate, state["cost"]),
        "grad": _tree_select(accepted, grad_candidate, state["grad"]),
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
    if _host_bool(success):
        return "converged"
    if int(_host_scalar(status, dtype=np.int64)) == 2:
        return "non-finite residual, gradient, or linear solve encountered"
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
    """Host-driven Levenberg-Marquardt solver for least-squares residuals.

    The LM loop is matrix-free: it uses ``jvp``/``vjp`` products inside GMRES
    and only rebuilds the dense residual Jacobian/Hessian once at the final
    iterate so existing Boozer adjoint consumers retain their contract.
    """
    residual_eval = jax.jit(_flattened_residual_output(residual_fn))

    x = jax.tree_util.tree_map(jnp.asarray, x0)
    residual, cost, grad, grad_norm_inf, _ = _least_squares_gradient_state(
        residual_eval,
        x,
    )
    x_dtype = _require_tree_first_leaf(
        x,
        detail="Least-squares initial state must contain at least one leaf.",
    ).dtype
    damping = _lm_defaults(x_dtype)["initial_damping"]
    status = 1
    success = bool(grad_norm_inf <= tol)
    nit = 0

    while nit < maxiter and not success:
        step_state = _lm_iteration(
            residual_eval,
            {
                "x": x,
                "residual": residual,
                "cost": cost,
                "grad": grad,
                "grad_norm_inf": grad_norm_inf,
                "damping": damping,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
                "status": jnp.asarray(status, dtype=jnp.int32),
                "accepted": jnp.asarray(False),
                "success": jnp.asarray(False),
            },
            tol=jnp.asarray(tol, dtype=x_dtype),
        )
        nit = int(step_state["nit"])
        status = int(step_state["status"])
        damping = step_state["damping"]
        if bool(step_state["accepted"]):
            x = step_state["x"]
            residual = step_state["residual"]
            cost = step_state["cost"]
            grad = step_state["grad"]
            grad_norm_inf = step_state["grad_norm_inf"]
            if callback is not None:
                callback(_hostify_optimizer_tree(x))
            if progress_callback is not None:
                progress_callback(nit, float(cost), float(grad_norm_inf))
        success = bool(step_state["success"])
        if status == 2:
            break

    residual, residual_jacobian, _flat_grad, hessian = (
        _materialize_dense_least_squares_linearization(residual_eval, x)
    )

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
    args=(),
):
    """Trace-safe Levenberg-Marquardt solver for least-squares residuals."""
    runner = _make_traceable_levenberg_marquardt_runner(
        residual_fn,
        int(maxiter),
        float(tol),
        callback,
        progress_callback,
    )
    return runner(x0, _normalize_solver_args(args))


# ---------------------------------------------------------------------------
# Newton solvers (public path, no jax._src)
# ---------------------------------------------------------------------------


def _materialize_dense_linear_operator(linear_operator_fn, x):
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    cols = lax.map(lambda basis: linear_operator_fn(x, basis), eye)
    return jnp.swapaxes(cols, 0, 1)


def _hessian_vector_product_fn(objective_fn):
    grad_fn = jax.grad(objective_fn)
    return jax.jit(lambda x, v: jax.jvp(grad_fn, (x,), (v,))[1])


def _jacobian_vector_product_fn(residual_fn):
    return jax.jit(lambda x, v: jax.jvp(residual_fn, (x,), (v,))[1])


def _materialize_dense_hessian(hvp_fn, x):
    dense = _materialize_dense_linear_operator(hvp_fn, x)
    return 0.5 * (dense + dense.T)


def _materialize_dense_jacobian(jvp_fn, x):
    return _materialize_dense_linear_operator(jvp_fn, x)


def _dense_operator_nbytes(rows, cols, dtype):
    return int(rows) * int(cols) * np.dtype(dtype).itemsize


def _dense_operator_exceeds_bytes_limit(rows, cols, dtype, max_dense_bytes):
    if max_dense_bytes is None:
        return False
    return _dense_operator_nbytes(rows, cols, dtype) > int(max_dense_bytes)


def _exact_newton_dense_jacobian_report(rows, cols, dtype, max_dense_bytes):
    return {
        "dense_jacobian_shape": (int(rows), int(cols)),
        "dense_jacobian_bytes": _dense_operator_nbytes(rows, cols, dtype),
        "max_dense_jacobian_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _exact_newton_dense_jacobian_message(rows, cols, dtype, max_dense_bytes):
    required_bytes = _dense_operator_nbytes(rows, cols, dtype)
    return (
        "Exact Newton skipped dense Jacobian materialization because "
        f"the final {int(rows)}x{int(cols)} Jacobian in dtype {np.dtype(dtype)} "
        f"would require {required_bytes} bytes, exceeding "
        f"max_dense_jacobian_bytes={int(max_dense_bytes)}."
    )


def _exact_newton_dense_jacobian_policy(rows, cols, dtype, max_dense_bytes):
    report = _exact_newton_dense_jacobian_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    materialize_jacobian = not _dense_operator_exceeds_bytes_limit(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_jacobian:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_jacobian_finalization"
        report["message"] = _exact_newton_dense_jacobian_message(
            rows,
            cols,
            dtype,
            max_dense_bytes,
        )
    return materialize_jacobian, report


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


def _gmres_solve_exact_newton_system(jvp_fn, x, rhs, *, tol):
    n = rhs.shape[0]
    restart = max(5, min(n, 50))
    maxiter = max(10, min(4 * n, 200))

    def matvec(v):
        return jvp_fn(x, v)

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
    val_and_grad_fn = _cached_jit_value_and_grad(objective_fn)
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


@lru_cache(maxsize=128)
def _make_traceable_newton_polish_runner(
    objective_fn,
    maxiter,
    tol,
    stab,
    progress_callback,
):
    def run_solver(x_init, fn_args):
        def objective_eval(x):
            return objective_fn(x, *fn_args)

        grad_fn = jax.grad(objective_eval)
        val_and_grad_fn = jax.value_and_grad(objective_eval)

        def hvp_fn(x, v):
            return jax.jvp(grad_fn, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = jnp.asarray(tol, dtype=dtype)
        linear_tol = jnp.minimum(
            jnp.asarray(1e-10, dtype=dtype),
            jnp.maximum(
                tol_value * jnp.asarray(0.1, dtype=dtype),
                jnp.asarray(1e-14, dtype=dtype),
            ),
        )
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

    return jax.jit(run_solver)


def newton_polish_traceable(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    progress_callback=None,
    args=(),
):
    """Trace-safe Newton polish for JAX-traceable objective paths.

    This variant keeps all loop state and fallback decisions inside JAX control
    flow so higher-level traced objectives can invoke the Newton stage without
    crossing back into Python.
    """
    runner = _make_traceable_newton_polish_runner(
        objective_fn,
        int(maxiter),
        float(tol),
        float(stab),
        progress_callback,
    )
    return runner(x0, _normalize_solver_args(args))


def newton_exact(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
    max_dense_jacobian_bytes=None,
):
    """Newton solver for the exact Boozer residual system ``r(x) = 0``.

    Iterations solve the linearized system with GMRES against exact
    Jacobian-vector products, avoiding dense Jacobian materialization in the
    hot loop. The dense Jacobian is rebuilt once at the final iterate so the
    existing Boozer PLU / adjoint contract stays intact.
    """
    res_fn = jax.jit(residual_fn)
    jvp_fn = _jacobian_vector_product_fn(residual_fn)

    x = x0
    r = res_fn(x)
    norm = jnp.linalg.norm(r)
    linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))

    nit = 0
    while nit < maxiter and float(norm) > tol:
        dx, linear_residual, _ = _gmres_solve_exact_newton_system(
            jvp_fn,
            x,
            r,
            tol=linear_tol,
        )
        linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
        if np.all(np.isfinite(np.asarray(dx))) and linear_residual_norm > linear_tol:
            correction, _, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                x,
                linear_residual,
                tol=linear_tol,
            )
            if np.all(np.isfinite(np.asarray(correction))):
                dx = dx + correction
        x = x - dx
        r = res_fn(x)
        norm = jnp.linalg.norm(r)
        nit += 1

    rows = int(np.prod(np.shape(r)))
    cols = int(np.prod(np.shape(x)))
    materialize_jacobian, report = _exact_newton_dense_jacobian_policy(
        rows,
        cols,
        x.dtype,
        max_dense_jacobian_bytes,
    )
    if not materialize_jacobian:
        return {
            "x": x,
            "residual": r,
            "jacobian": None,
            "nit": nit,
            "success": bool(float(norm) <= tol),
            "jacobian_materialized": False,
            **report,
        }

    J = _materialize_dense_jacobian(jvp_fn, x)

    return {
        "x": x,
        "residual": r,
        "jacobian": J,
        "nit": nit,
        "success": bool(float(norm) <= tol),
        "jacobian_materialized": True,
        **report,
    }


@lru_cache(maxsize=128)
def _make_traceable_exact_newton_runner(
    residual_fn,
    maxiter,
    tol,
    materialize_jacobian,
):
    def run_solver(x_init, fn_args):
        def residual_eval(x):
            return residual_fn(x, *fn_args)

        def jvp_fn(x, v):
            return jax.jvp(residual_eval, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = jnp.asarray(tol, dtype=dtype)
        linear_tol = jnp.minimum(
            jnp.asarray(1e-10, dtype=dtype),
            jnp.maximum(
                tol_value * jnp.asarray(0.1, dtype=dtype),
                jnp.asarray(1e-14, dtype=dtype),
            ),
        )
        r0 = residual_eval(x_init)
        norm0 = jnp.linalg.norm(r0)

        def cond_fun(state):
            return (state["nit"] < maxiter) & (state["norm"] > tol_value)

        def body_fun(state):
            dx, linear_residual, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                state["x"],
                state["residual"],
                tol=linear_tol,
            )
            linear_residual_norm = jnp.linalg.norm(linear_residual)

            def add_correction(current_dx):
                correction, _, _ = _gmres_solve_exact_newton_system(
                    jvp_fn,
                    state["x"],
                    linear_residual,
                    tol=linear_tol,
                )
                return lax.cond(
                    jnp.all(jnp.isfinite(correction)),
                    lambda corr: current_dx + corr,
                    lambda _corr: current_dx,
                    correction,
                )

            dx = lax.cond(
                jnp.all(jnp.isfinite(dx)) & (linear_residual_norm > linear_tol),
                add_correction,
                lambda current_dx: current_dx,
                dx,
            )
            x_next = state["x"] - dx
            residual_next = residual_eval(x_next)
            return {
                "x": x_next,
                "residual": residual_next,
                "norm": jnp.linalg.norm(residual_next),
                "nit": state["nit"] + 1,
            }

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "residual": r0,
                "norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
            },
        )
        result = {
            "x": state["x"],
            "residual": state["residual"],
            "nit": state["nit"],
            "success": state["norm"] <= tol_value,
        }
        if materialize_jacobian:
            result["jacobian"] = _materialize_dense_jacobian(jvp_fn, state["x"])
        return result

    return jax.jit(run_solver)


def newton_exact_traceable(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
    args=(),
    max_dense_jacobian_bytes=None,
):
    """Trace-safe Newton solver for the exact Boozer residual system.

    The loop keeps Jacobian application matrix-free via JVPs and materializes a
    dense Jacobian only once at the final iterate for downstream LU-based
    contracts.
    """
    normalized_args = _normalize_solver_args(args)
    x0_shape = tuple(int(dim) for dim in np.shape(x0))
    residual_shape = jax.eval_shape(
        lambda x, fn_args: residual_fn(x, *fn_args),
        x0,
        normalized_args,
    ).shape
    rows = int(np.prod(residual_shape))
    cols = int(np.prod(x0_shape))
    x0_dtype = np.dtype(x0.dtype) if hasattr(x0, "dtype") else np.result_type(x0)
    materialize_jacobian, report = _exact_newton_dense_jacobian_policy(
        rows,
        cols,
        x0_dtype,
        max_dense_jacobian_bytes,
    )
    runner = _make_traceable_exact_newton_runner(
        residual_fn,
        int(maxiter),
        float(tol),
        materialize_jacobian,
    )
    result = runner(x0, normalized_args)
    if materialize_jacobian:
        result["jacobian_materialized"] = True
        result.update(report)
        return result
    result["jacobian"] = None
    result["jacobian_materialized"] = False
    result.update(report)
    return result


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
    """Explicit CPU/reference least-squares entrypoint."""
    return _load_reference_optimizer_module().reference_least_squares(
        residual_fn,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        callback=callback,
        progress_callback=progress_callback,
    )


def target_least_squares(
    residual_fn,
    x0,
    *,
    method="lm-ondevice",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Explicit JAX target least-squares entrypoint."""
    if method != "lm-ondevice":
        raise ValueError(
            "target_least_squares() only supports method='lm-ondevice'. "
            f"Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")
    result = levenberg_marquardt_traceable(
        residual_fn,
        x0,
        maxiter=maxiter,
        tol=tol,
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
    )

    nit = int(_host_scalar(result["nit"], dtype=np.int64))
    status = int(_host_scalar(result["status"], dtype=np.int64))
    success = _host_bool(result["success"])
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
        message=_least_squares_result_message(
            status,
            success,
        ),
    )


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
    """Compatibility least-squares entrypoint that dispatches by lane."""
    if method not in _SUPPORTED_LEAST_SQUARES_METHODS:
        raise ValueError(
            "Unknown least-squares method "
            f"{method!r}. Supported: {sorted(_SUPPORTED_LEAST_SQUARES_METHODS)}."
        )
    if method == "lm":
        _raise_if_target_lane_required(
            component="optimizer_jax.jax_least_squares",
            method=method,
            detail=_STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
        )
    if method == "lm":
        return reference_least_squares(
            residual_fn,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
            callback=callback,
            progress_callback=progress_callback,
        )
    return target_least_squares(
        residual_fn,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        callback=callback,
        progress_callback=progress_callback,
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
    """Explicit CPU/reference scalar optimizer entrypoint."""
    if method in _REFERENCE_JAX_METHODS:
        _raise_if_target_lane_required(
            component="optimizer_jax.reference_minimize",
            method=method,
            detail=_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL,
        )
        _raise_if_strict_optimizer_fallback(
            component="optimizer_jax.reference_minimize",
            method=method,
            detail=_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL,
        )
        result = adam_optimize(
            fun,
            x0,
            value_and_grad=value_and_grad,
            maxiter=maxiter,
            tol=tol,
            options=options,
            callback=callback,
            progress_callback=progress_callback,
        )
        return _adam_result_to_optimize_result(result)
    return _load_reference_optimizer_module().reference_minimize(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        value_and_grad=value_and_grad,
        callback=callback,
        progress_callback=progress_callback,
    )


def target_minimize(
    fun,
    x0,
    *,
    method="bfgs-ondevice",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
    failure_callback=None,
):
    """Explicit JAX target scalar optimizer entrypoint."""
    if method in _TARGET_PUBLIC_METHODS:
        require_target_backend_x64("ondevice")
        result = adam_optimize_traceable(
            fun,
            x0,
            value_and_grad=value_and_grad,
            maxiter=maxiter,
            tol=tol,
            options=options,
            callback=callback,
            progress_callback=progress_callback,
        )
        return _adam_result_to_optimize_result(result)

    if method not in _TARGET_PRIVATE_METHODS:
        raise ValueError(
            "target_minimize() only supports target-lane methods "
            f"{sorted(_TARGET_METHODS)}. Got {method!r}."
        )

    pytree_adapter = _prepare_optimizer_pytree_adapter(x0)

    def finalize(result):
        return _finalize_optimizer_result(result, pytree_adapter)

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    if failure_callback is not None:
        options["failure_callback"] = failure_callback

    require_target_backend_x64("ondevice")

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
            failure_callback=options.get("failure_callback"),
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
            failure_callback=options.get("failure_callback"),
        )
        return finalize(_private_lbfgs_result_to_optimize_result(state))
    raise ValueError(f"Unknown target optimizer method {method!r}.")


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
    """Compatibility scalar optimizer entrypoint that dispatches by lane."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )

    if method in _REFERENCE_METHODS | _REFERENCE_JAX_METHODS:
        detail = (
            _STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL
            if method in _REFERENCE_JAX_METHODS
            else _STRICT_REFERENCE_OPTIMIZER_DETAIL
        )
        _raise_if_target_lane_required(
            component="optimizer_jax.jax_minimize",
            method=method,
            detail=detail,
        )
        return reference_minimize(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
            value_and_grad=value_and_grad,
            callback=callback,
            progress_callback=progress_callback,
        )
    return target_minimize(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        value_and_grad=value_and_grad,
        callback=callback,
        progress_callback=progress_callback,
    )
