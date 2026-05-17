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
  - ``method="lm-minpack-ondevice"``: trace-safe dense-QR
    Levenberg-Marquardt for residual-vector objectives on the target lane.
  - ``method="optimistix-lm-ondevice"``: optional Optimistix
    Levenberg-Marquardt lane with a Lineax LSMR inner solve.

LM family note:
  Neither ``"lm"`` nor ``"lm-ondevice"`` is a port of MINPACK ``lmder``
  (the algorithm behind ``scipy.optimize.least_squares(method="lm")``).
  Both methods route through ``levenberg_marquardt`` /
  ``levenberg_marquardt_traceable`` (host-driven and trace-safe variants
  of the same JAX LM loop). They are **algorithmically distinct** from
  MINPACK along three load-bearing axes:

  - **Inner solve.** MINPACK uses a pivoted-QR factorization of the
    Jacobian; the JAX LM uses matrix-free GMRES against the
    regularized Gauss-Newton operator ``J^T J + λI`` (no QR pivoting,
    no dense Jacobian factorization in the inner step). See
    ``_lm_iteration`` and ``_gmres_solve_least_squares_system``.
  - **Termination.** MINPACK terminates on independent ``ftol``, ``xtol``,
    and ``gtol`` criteria. The JAX LM now surfaces the matrix-free-computable
    subset as ``info`` codes 1, 2, 3, 5, 6, and 7. It also supports ``gtol``
    as the matrix-free infinity-norm gradient gate when callers explicitly
    provide it; otherwise the legacy ``tol`` gradient gate is preserved.
    MINPACK ``info`` codes 4 and 8 both require the pivoted-QR scaled gradient
    norm and remain outside this matrix-free lane.
  - **Damping update.** MINPACK uses Marquardt's classic
    expand/contract scaling; the JAX LM uses the same symmetric damping
    factors for this matrix-free lane — decrease ``× 0.5`` on
    ``ratio > 0.75`` and increase ``× 2.0`` on ``ratio < 0.25`` or rejected
    steps (see ``_lm_iteration`` and ``_lm_defaults``).

  The opt-in ``"lm-minpack-ondevice"`` lane uses a dense pivoted-QR
  augmented-system step, so it matches MINPACK's QR conditioning model at
  tolerance level without claiming MINPACK's packed-QR byte identity.

  The opt-in ``"optimistix-lm-ondevice"`` lane delegates the nonlinear
  least-squares loop to Optimistix and the inner linear solves to Lineax LSMR.
  It is tolerance-equivalent to the in-tree JAX LM family, not a MINPACK
  parity lane, and requires the optional ``JAX_OPTIMISTIX`` dependency extra.

  Consequence: the JAX LM lanes are **tolerance-equivalent** to MINPACK
  ``lmder`` on well-conditioned fixtures but **not byte-equivalent**;
  ``"lm"`` (reference, host-driven) and ``"lm-ondevice"`` (target,
  trace-safe) are each other's byte-equality oracle, not MINPACK.
  Callers needing MINPACK byte-equality must invoke
  ``scipy.optimize.least_squares(method="lm")`` directly. Use
  ``optimizer_backend="ondevice"`` + ``least_squares_algorithm="lm"``
  to engage the matrix-free on-device LM lane, or
  ``optimizer_backend="ondevice"`` + ``least_squares_algorithm="lm-minpack"``
  to engage the dense pivoted-QR LM lane, or
  ``optimizer_backend="ondevice"`` +
  ``least_squares_algorithm="optimistix-lm"`` to engage the optional
  Optimistix/Lineax LSMR lane.

Target private methods (minimum supported JAX floor 0.9.2):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: JAX on-device L-BFGS.

Target SciPy-control method:
  - ``method="lbfgs-scipy-jax"``: host SciPy L-BFGS-B control with JAX
    target-lane value/grad evaluations.
  - ``method="lbfgs-scipy-jax-fullgraph"``: host SciPy L-BFGS-B control with
    JAX value/grad evaluations over a caller-owned full Optimizable graph.

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
from functools import lru_cache, wraps
import re
from threading import Lock
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.flatten_util import ravel_pytree
from jax import lax
from jax.scipy.sparse.linalg import gmres
import scipy.linalg
from scipy.optimize import OptimizeResult

from ..backend import (
    get_backend_config,
    raise_if_strict_jax_fallback,
    strict_target_lane_purity,
    target_lane_purity_requested,
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
    "VALID_OUTER_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "jax_least_squares",
    "jax_least_squares_optimistix",
    "jax_minimize",
    "levenberg_marquardt",
    "levenberg_marquardt_minpack_traceable",
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
    "wrap_strict_target_lane_value_and_grad",
    "target_least_squares",
    "target_minimize",
]


PRIVATE_OPTIMIZER_JAX_VERSION = "0.9.2"
VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})
VALID_OUTER_OPTIMIZER_BACKENDS = frozenset(
    {"scipy", "ondevice", "scipy-jax", "scipy-jax-fullgraph"}
)
OPTIMIZER_BACKEND_ROLE = {
    "scipy": "reference",
    "ondevice": "target",
    "scipy-jax": "target-scipy-control",
    "scipy-jax-fullgraph": "target-scipy-control-fullgraph",
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = frozenset(
    {"ondevice", "scipy-jax", "scipy-jax-fullgraph"}
)
VALID_LEAST_SQUARES_ALGORITHMS = frozenset(
    {"quasi-newton", "lm", "lm-minpack", "optimistix-lm"}
)
_SUPPORTED_METHODS = {
    "adam",
    "adam-ondevice",
    "bfgs",
    "lbfgs",
    "lbfgs-scipy-jax",
    "lbfgs-scipy-jax-fullgraph",
    "lbfgs-trace",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
_TARGET_LEAST_SQUARES_METHODS = frozenset(
    {"lm-ondevice", "lm-minpack-ondevice", "optimistix-lm-ondevice"}
)
_SUPPORTED_LEAST_SQUARES_METHODS = frozenset({"lm"}) | _TARGET_LEAST_SQUARES_METHODS
_RESIDUAL_LEAST_SQUARES_ALGORITHMS = frozenset({"lm", "lm-minpack", "optimistix-lm"})
_DEFAULT_LM_FTOL = 1e-8
_DEFAULT_LM_XTOL = 1e-8
_OPTIMISTIX_LM_DEFAULT_FTOL = _DEFAULT_LM_FTOL
_OPTIMISTIX_LM_DEFAULT_XTOL = _DEFAULT_LM_XTOL
_OPTIMISTIX_LM_DEFAULT_GTOL = None
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})
_REFERENCE_TRACE_METHODS = frozenset({"lbfgs-trace"})
_REFERENCE_JAX_METHODS = frozenset({"adam"})
_TARGET_PRIVATE_METHODS = frozenset({"bfgs-ondevice", "lbfgs-ondevice"})
_TARGET_SCIPY_CONTROL_METHODS = frozenset(
    {"lbfgs-scipy-jax", "lbfgs-scipy-jax-fullgraph"}
)
_TARGET_PUBLIC_METHODS = frozenset({"adam-ondevice"})
_TARGET_METHODS = (
    _TARGET_PRIVATE_METHODS | _TARGET_PUBLIC_METHODS | _TARGET_SCIPY_CONTROL_METHODS
)
_TARGET_LBFGSB_METHODS = frozenset({"lbfgs-ondevice"}) | _TARGET_SCIPY_CONTROL_METHODS
_UNSUPPORTED_TARGET_LBFGSB_OPTIONS = frozenset({"initial_step_size", "maxgrad"})
_STRICT_REFERENCE_OPTIMIZER_DETAIL = "the host-side SciPy reference optimizer lane"
_STRICT_REFERENCE_JAX_OPTIMIZER_DETAIL = "the host-side JAX reference optimizer lane"
_STRICT_REFERENCE_LEAST_SQUARES_DETAIL = (
    "the host-side reference least-squares optimizer lane"
)
_EISENSTAT_WALKER_GAMMA = 0.9
# α=2 is inlined as ``ratio * ratio`` inside
# ``_eisenstat_walker_choice2_tolerance`` for bit-stable evaluation; see
# Eisenstat & Walker (1996) eq. (2.6).
_EISENSTAT_WALKER_MIN_ETA = 1.0e-12
_EISENSTAT_WALKER_MAX_ETA = 0.5
_NEWTON_BACKTRACKING_MAX_STEPS = 8
_HAGER_HIGHAM_CONDITION_ITERATIONS = 5
_SCALAR_VALUE_AND_GRAD_CACHE_LOCK = Lock()
_CACHEABLE_VALUE_AND_GRAD_ATTR = "_simsopt_cache_jit_value_and_grad"
_CACHED_VALUE_AND_GRAD_ATTR = "_simsopt_cached_jit_value_and_grad"
_STRUCTURED_SOLVER_CACHE_TOKEN_ATTR = "_simsopt_structured_solver_cache_token"


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
    leaf_signature: tuple[tuple[tuple[int, ...], str], ...]

    def _hostify_flat(self, flat_x, *, dtype=None):
        return _hostify_optimizer_tree(
            self.unravel(_optimizer_flat_vector(flat_x, dtype=dtype))
        )

    def wrap_fun(self, fun, *, value_and_grad: bool):
        if not value_and_grad:

            def wrapped_fun(flat_x):
                return fun(self.unravel(_optimizer_flat_vector(flat_x)))

            return wrapped_fun

        def wrapped_value_and_grad(flat_x):
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

        return wrapped_value_and_grad

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

    def solver_cache_key(self) -> tuple[object, ...]:
        return (
            self.flat_dtype.str,
            repr(self.tree_def),
            self.leaf_signature,
        )


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


def _require_native_cpu_reference_backend_for_trace_adapter(
    *,
    component: str,
    method: str,
) -> None:
    backend_config = get_backend_config()
    if backend_config.backend != "jax":
        return
    raise RuntimeError(
        f"{component} cannot use the CPU/C++ trace adapter for method={method!r} "
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


def _optimizer_shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return shape
    return np.shape(value)


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


_FLAT_OPTIMIZER_REJECTED_DTYPE_KINDS = frozenset("MmOSU")


def _is_supported_flat_optimizer_array(value) -> bool:
    dtype = value.dtype
    dtype_kind = getattr(dtype, "kind", None)
    if dtype_kind in _FLAT_OPTIMIZER_REJECTED_DTYPE_KINDS:
        return False
    return value.ndim == 1 and (
        jax.dtypes.issubdtype(dtype, jnp.number)
        or jax.dtypes.issubdtype(dtype, jnp.bool_)
    )


def _is_flat_optimizer_vector(x0) -> bool:
    if isinstance(x0, (jax.Array, np.ndarray)):
        return _is_supported_flat_optimizer_array(x0)
    if isinstance(x0, (list, tuple)):
        if not all(isinstance(item, (int, float, np.generic)) for item in x0):
            return False
        return _is_supported_flat_optimizer_array(np.asarray(x0))
    return False


def _prepare_optimizer_pytree_adapter(x0):
    if _is_flat_optimizer_vector(x0):
        return None
    leaves, tree_def = jax.tree_util.tree_flatten(x0)
    flat_x0, unravel = ravel_pytree(x0)
    flat_dtype = np.dtype(_optimizer_dtype(flat_x0))
    leaf_signature = tuple(
        (
            tuple(int(dim) for dim in _optimizer_shape(leaf)),
            np.dtype(_optimizer_dtype(leaf)).str,
        )
        for leaf in leaves
    )
    return _OptimizerPytreeAdapter(
        flat_dtype=flat_dtype,
        unravel=unravel,
        tree_def=tree_def,
        leaf_signature=leaf_signature,
    )


def _mark_cacheable_jit_value_and_grad(fun):
    # ``fun`` must be a Python callable that accepts ``setattr`` (def, lambda,
    # closure). Production call sites pass only such callables; if a future
    # caller passes a builtin or ``__slots__`` instance the ``AttributeError``
    # surfaces the contract violation rather than silently no-op'ing.
    setattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    return fun


def _mark_structured_private_solver_cacheable(fun, *, cache_token):
    # Same contract as ``_mark_cacheable_jit_value_and_grad``.
    setattr(fun, _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR, cache_token)
    return fun


def wrap_strict_target_lane_value_and_grad(fun):
    """Wrap target-lane value/grad calls in the stack-scoped purity guard."""
    if not target_lane_purity_requested():
        return fun

    @wraps(fun)
    def wrapped(*args, **kwargs):
        with strict_target_lane_purity():
            return fun(*args, **kwargs)

    return wrapped


def _cached_jit_value_and_grad(fun):
    if not getattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, False):
        return jax.jit(jax.value_and_grad(fun))
    cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
    if cached is not None:
        return cached
    compiled = jax.jit(jax.value_and_grad(fun))
    # Double-checked install under the cache lock. ``fun`` has already
    # been marked via ``_mark_cacheable_jit_value_and_grad`` (the marker
    # check above gated this branch), so ``setattr`` cannot raise.
    with _SCALAR_VALUE_AND_GRAD_CACHE_LOCK:
        cached = getattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, None)
        if cached is not None:
            return cached
        setattr(fun, _CACHED_VALUE_AND_GRAD_ATTR, compiled)
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
_private_pkg = None

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
            import simsopt.geo.optimizer_jax_private as optimizer_jax_private
        except ModuleNotFoundError as exc:
            if exc.name != "simsopt.geo.optimizer_jax_private":
                raise
            _private_pkg = False
        else:
            _private_pkg = optimizer_jax_private
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
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "optimizer_backend must be one of: scipy, ondevice, scipy-jax, "
            "scipy-jax-fullgraph."
        )
    if optimizer_backend == "scipy":
        return resolve_reference_optimizer_method(limited_memory=limited_memory)
    if optimizer_backend == "scipy-jax":
        return "lbfgs-scipy-jax"
    if optimizer_backend == "scipy-jax-fullgraph":
        return "lbfgs-scipy-jax-fullgraph"
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
            f"least_squares_algorithm={least_squares_algorithm!r} is incompatible "
            "with limited_memory=True."
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
            f"least_squares_algorithm={least_squares_algorithm!r} is incompatible "
            "with limited_memory=True."
        )
    if least_squares_algorithm == "lm-minpack":
        return "lm-minpack-ondevice"
    if least_squares_algorithm == "optimistix-lm":
        return "optimistix-lm-ondevice"
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
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "optimizer_backend must be one of: scipy, ondevice, scipy-jax, "
            "scipy-jax-fullgraph."
        )
    if field_backend == "jax":
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice', optimizer_backend='scipy-jax', or "
            "optimizer_backend='scipy-jax-fullgraph'. "
            "The SciPy/reference optimizer lane is CPU/reference-only."
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
    if optimizer_backend not in VALID_OUTER_OPTIMIZER_BACKENDS:
        raise ValueError(
            "optimizer_backend must be one of: scipy, ondevice, scipy-jax, "
            "scipy-jax-fullgraph."
        )
    if field_backend != "jax" or optimizer_backend not in {
        "ondevice",
        "scipy-jax",
        "scipy-jax-fullgraph",
    }:
        raise ValueError(
            f"{component_label} with backend='jax' requires "
            "optimizer_backend='ondevice', optimizer_backend='scipy-jax', or "
            "optimizer_backend='scipy-jax-fullgraph'. "
            "The SciPy/reference optimizer lane is CPU/reference-only."
        )
    require_target_backend_x64(optimizer_backend)
    if optimizer_backend in {"scipy-jax", "scipy-jax-fullgraph"}:
        if least_squares_algorithm != "quasi-newton":
            raise ValueError(
                f"optimizer_backend={optimizer_backend!r} only supports "
                "least_squares_algorithm='quasi-newton'."
            )
        method = resolve_optimizer_backend_method(
            optimizer_backend,
            limited_memory=limited_memory,
        )
        return TargetOptimizerContract(
            method=method,
            use_least_squares_objective=False,
        )
    method = resolve_target_least_squares_optimizer_method(
        limited_memory=limited_memory,
        least_squares_algorithm=least_squares_algorithm,
    )
    return TargetOptimizerContract(
        method=method,
        use_least_squares_objective=method in _TARGET_LEAST_SQUARES_METHODS,
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
    limited_memory = least_squares_algorithm not in _RESIDUAL_LEAST_SQUARES_ALGORITHMS
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


def _tree_l2_norm(tree):
    return jnp.sqrt(jnp.maximum(_tree_vdot_real(tree, tree), _device_scalar(0.0)))


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

    run_solver.__name__ = "adam_traceable_run_solver"
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
    ftol,
    xtol,
    gtol,
    materialize_dense_linearization,
    max_dense_linearization_bytes,
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
        gradient_tol = _lm_gradient_tol(tol, gtol, dtype=x_dtype)
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
            "delta": _lm_initial_delta(x_init, dtype=x_dtype),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "info": jnp.asarray(0, dtype=jnp.int32),
            "accepted": jnp.asarray(False),
            "success": grad_norm_inf0 <= gradient_tol,
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (~state["success"])
                & (state["status"] != 2)
                & (state["info"] == 0)
            )

        def body_fun(state):
            next_state = _lm_iteration(
                residual_eval,
                state,
                tol=tol_value,
                gradient_tol=gradient_tol,
                ftol=_device_scalar(ftol, dtype=x_dtype),
                xtol=_device_scalar(xtol, dtype=x_dtype),
                maxiter=maxiter,
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
        residual_final = residual_eval(state["x"])
        linearization_rows = int(np.asarray(jnp.asarray(residual_final).size))
        linearization_cols = sum(
            int(np.asarray(jnp.asarray(leaf).size))
            for leaf in jax.tree_util.tree_leaves(state["x"])
        )
        materialize_linearization = bool(materialize_dense_linearization)
        dense_report = _least_squares_dense_linearization_report(
            linearization_rows,
            linearization_cols,
            x_dtype,
            max_dense_linearization_bytes,
        )
        dense_report["failure_category"] = None
        dense_report["failure_stage"] = None
        dense_report["message"] = None
        residual_jacobian = None
        hessian = None
        if materialize_linearization:
            materialize_linearization, dense_report = (
                _least_squares_dense_linearization_policy(
                    linearization_rows,
                    linearization_cols,
                    x_dtype,
                    max_dense_linearization_bytes,
                )
            )
            if materialize_linearization:
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
            "info": state["info"],
            "success": state["success"],
            "dense_linearization_materialized": materialize_linearization,
            **dense_report,
        }

    run_solver.__name__ = "traceable_levenberg_marquardt_run_solver"
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
        "initial_delta_factor": _device_scalar(100.0, dtype=dtype),
        "accept_threshold": _device_scalar(1.0e-4, dtype=dtype),
        "increase_factor": _device_scalar(2.0, dtype=dtype),
        "decrease_factor": _device_scalar(0.5, dtype=dtype),
        "minimum_delta_update": _device_scalar(0.1, dtype=dtype),
        "ratio_low": _device_scalar(0.25, dtype=dtype),
        "ratio_high": _device_scalar(0.75, dtype=dtype),
        "predicted_floor": _device_scalar(1.0e-18, dtype=dtype),
    }


def _lm_initial_delta(x, *, dtype):
    x_norm = _tree_l2_norm(x)
    return _lm_defaults(dtype)["initial_delta_factor"] * jnp.maximum(
        x_norm,
        _device_scalar(1.0, dtype=dtype),
    )


def _lm_gradient_tol(tol, gtol, *, dtype):
    if gtol is None:
        return _optimizer_scalar(tol, dtype=dtype)
    return _optimizer_scalar(gtol, dtype=dtype)


def _optimistix_lm_nondefault_tuning_options(ftol, xtol, gtol):
    unsupported = []
    if ftol is None or float(ftol) != _OPTIMISTIX_LM_DEFAULT_FTOL:
        unsupported.append("ftol")
    if xtol is None or float(xtol) != _OPTIMISTIX_LM_DEFAULT_XTOL:
        unsupported.append("xtol")
    if gtol is not None:
        unsupported.append("gtol")
    return tuple(unsupported)


def _require_optimistix_lm_contract_options(
    *,
    ftol,
    xtol,
    gtol,
    callback,
    progress_callback,
):
    if callback is not None or progress_callback is not None:
        raise ValueError(
            "optimistix-lm-ondevice does not support solver callbacks. "
            "Use method='lm-ondevice' for callback-instrumented LM runs."
        )
    unsupported = _optimistix_lm_nondefault_tuning_options(ftol, xtol, gtol)
    if unsupported:
        unsupported_options = ", ".join(unsupported)
        raise ValueError(
            "optimistix-lm-ondevice uses a single tol value for Optimistix "
            "and Lineax convergence. Non-default LM tuning option(s) are not "
            f"supported: {unsupported_options}."
        )


def _matrix_free_lm_info(
    *,
    actual_reduction,
    predicted_reduction,
    cost,
    delta,
    x_norm,
    nit,
    maxiter,
    ftol,
    xtol,
    epsmch,
):
    """Return MINPACK-style info for the matrix-free-computable LM subset."""
    one = jnp.asarray(1.0, dtype=cost.dtype)
    half = jnp.asarray(0.5, dtype=cost.dtype)
    cost_floor = jnp.maximum(cost, jnp.finfo(cost.dtype).tiny)
    nonnegative_reduction = actual_reduction >= jnp.asarray(0.0, dtype=cost.dtype)
    relative_actual = actual_reduction / cost_floor
    relative_predicted = predicted_reduction / cost_floor
    ratio = actual_reduction / jnp.maximum(predicted_reduction, cost_floor * epsmch)
    ftol_met = (
        nonnegative_reduction
        & (relative_actual <= ftol)
        & (relative_predicted <= ftol)
        & (half * ratio <= one)
    )
    xtol_met = delta <= xtol * x_norm

    info = jnp.asarray(0, dtype=jnp.int32)
    info = jnp.where(ftol_met, jnp.asarray(1, dtype=jnp.int32), info)
    info = jnp.where(xtol_met, jnp.asarray(2, dtype=jnp.int32), info)
    info = jnp.where(
        ftol_met & xtol_met,
        jnp.asarray(3, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (nit >= maxiter),
        jnp.asarray(5, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0)
        & nonnegative_reduction
        & (relative_actual <= epsmch)
        & (relative_predicted <= epsmch)
        & (half * ratio <= one),
        jnp.asarray(6, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (delta <= epsmch * x_norm),
        jnp.asarray(7, dtype=jnp.int32),
        info,
    )
    return info


def _lm_delta_after_step(delta, step_norm, ratio, actual_reduction, *, defaults):
    low_ratio_base = jnp.minimum(
        delta,
        step_norm / defaults["minimum_delta_update"],
    )
    low_ratio_scale = jnp.where(
        actual_reduction >= jnp.asarray(0.0, dtype=delta.dtype),
        defaults["decrease_factor"],
        defaults["minimum_delta_update"],
    )
    low_ratio_delta = low_ratio_scale * low_ratio_base
    updated_delta = jnp.where(
        ratio >= defaults["ratio_high"],
        step_norm / defaults["decrease_factor"],
        delta,
    )
    return jnp.where(
        ratio <= defaults["ratio_low"],
        low_ratio_delta,
        updated_delta,
    )


def _lm_iteration(flat_residual_fn, state, *, tol, gradient_tol, ftol, xtol, maxiter):
    state_dtype = _require_tree_first_leaf(
        state["x"],
        detail="Least-squares state x must contain at least one leaf.",
    ).dtype
    defaults = _lm_defaults(state_dtype)
    damping = _clip_lm_damping(state["damping"], dtype=state_dtype)
    linear_tol = jnp.minimum(
        _device_scalar(1.0e-10, dtype=state["cost"].dtype),
        jnp.maximum(
            _optimizer_scalar(tol, dtype=state["cost"].dtype)
            * _device_scalar(0.1, dtype=state["cost"].dtype),
            _device_scalar(1.0e-14, dtype=state["cost"].dtype),
        ),
    )
    _, current_pullback = jax.vjp(flat_residual_fn, state["x"])
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
    accepted = finite_candidate & (ratio >= defaults["accept_threshold"])
    step_norm = _tree_l2_norm(step)
    delta_after_step = _lm_delta_after_step(
        state["delta"],
        step_norm,
        jnp.asarray(ratio, dtype=state["delta"].dtype),
        jnp.asarray(actual_reduction, dtype=state["delta"].dtype),
        defaults=defaults,
    )

    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["decrease_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["increase_factor"],
            lambda __: damping,
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=state_dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["increase_factor"],
            dtype=state_dtype,
        ),
        operand=None,
    )
    x_next = _tree_select(accepted, x_candidate, state["x"])
    residual_next = lax.select(accepted, residual_candidate, state["residual"])
    cost_next = lax.select(accepted, cost_candidate, state["cost"])
    grad_next = _tree_select(accepted, grad_candidate, state["grad"])
    grad_norm_next = lax.select(
        accepted,
        grad_norm_candidate,
        state["grad_norm_inf"],
    )
    x_norm = _tree_l2_norm(x_next)
    next_nit = state["nit"] + 1
    info_candidate = _matrix_free_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=state["cost"],
        delta=delta_after_step,
        x_norm=x_norm,
        nit=next_nit,
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=state["cost"].dtype),
        xtol=jnp.asarray(xtol, dtype=state["cost"].dtype),
        epsmch=jnp.asarray(
            jnp.finfo(state["cost"].dtype).eps, dtype=state["cost"].dtype
        ),
    )
    info_next = lax.select(
        finite_candidate,
        info_candidate,
        jnp.asarray(0, dtype=jnp.int32),
    )
    legacy_success = grad_norm_next <= gradient_tol
    info_success = (info_next == 1) | (info_next == 2) | (info_next == 3)

    return {
        "x": x_next,
        "residual": residual_next,
        "cost": cost_next,
        "grad": grad_next,
        "grad_norm_inf": grad_norm_next,
        "damping": next_damping,
        "delta": delta_after_step,
        "nit": next_nit,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "info": info_next,
        "accepted": accepted,
        "success": finite_candidate & (legacy_success | info_success),
    }


def _least_squares_result_message(status, success, info=0):
    info_value = int(_host_scalar(info, dtype=np.int64))
    if info_value == 1:
        return "converged: ftol termination condition is satisfied"
    if info_value == 2:
        return "converged: xtol termination condition is satisfied"
    if info_value == 3:
        return "converged: both ftol and xtol termination conditions are satisfied"
    if info_value == 4:
        return "converged: gtol termination condition is satisfied"
    if info_value == 5:
        return "maximum iterations reached"
    if info_value == 6:
        return "ftol is too small; no further cost reduction is possible"
    if info_value == 7:
        return "xtol is too small; no further step reduction is possible"
    if info_value == 8:
        return "gtol is too small; residual is orthogonal to Jacobian columns"
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
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
):
    """Host-driven Levenberg-Marquardt solver for least-squares residuals.

    The LM loop is matrix-free: it uses ``jvp``/``vjp`` products inside GMRES
    and only rebuilds the dense residual Jacobian/Hessian once at the final
    iterate so existing Boozer adjoint consumers retain their contract.

    ``ftol`` and ``xtol`` feed the matrix-free MINPACK-style ``info`` subset.
    Explicit ``gtol`` values replace the legacy ``tol`` gradient gate with a
    matrix-free infinity-norm gradient gate. MINPACK's QR-scaled ``gtol``
    ``info`` code still requires pivoted-QR data and is therefore not emitted
    by this matrix-free solver.
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
    gradient_tol = _lm_gradient_tol(tol, gtol, dtype=x_dtype)
    damping = _lm_defaults(x_dtype)["initial_damping"]
    delta = _lm_initial_delta(x, dtype=x_dtype)
    status = 1
    success = bool(grad_norm_inf <= gradient_tol)
    info = 0
    nit = 0

    while nit < maxiter and not success and info == 0:
        step_state = _lm_iteration(
            residual_eval,
            {
                "x": x,
                "residual": residual,
                "cost": cost,
                "grad": grad,
                "grad_norm_inf": grad_norm_inf,
                "damping": damping,
                "delta": delta,
                "nit": jnp.asarray(nit, dtype=jnp.int32),
                "status": jnp.asarray(status, dtype=jnp.int32),
                "info": jnp.asarray(info, dtype=jnp.int32),
                "accepted": jnp.asarray(False),
                "success": jnp.asarray(False),
            },
            tol=_optimizer_scalar(tol, dtype=x_dtype),
            gradient_tol=gradient_tol,
            ftol=_optimizer_scalar(ftol, dtype=x_dtype),
            xtol=_optimizer_scalar(xtol, dtype=x_dtype),
            maxiter=int(maxiter),
        )
        nit = int(step_state["nit"])
        status = int(step_state["status"])
        info = int(step_state["info"])
        damping = step_state["damping"]
        delta = step_state["delta"]
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

    residual = residual_eval(x)
    linearization_rows = int(np.asarray(jnp.asarray(residual).size))
    linearization_cols = sum(
        int(np.asarray(jnp.asarray(leaf).size)) for leaf in jax.tree_util.tree_leaves(x)
    )
    dense_report = _least_squares_dense_linearization_report(
        linearization_rows,
        linearization_cols,
        x_dtype,
        max_dense_linearization_bytes,
    )
    dense_report["failure_category"] = None
    dense_report["failure_stage"] = None
    dense_report["message"] = None
    residual_jacobian = None
    hessian = None
    dense_linearization_materialized = bool(materialize_dense_linearization)
    if dense_linearization_materialized:
        dense_linearization_materialized, dense_report = (
            _least_squares_dense_linearization_policy(
                linearization_rows,
                linearization_cols,
                x_dtype,
                max_dense_linearization_bytes,
            )
        )
        if dense_linearization_materialized:
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
        "info": info,
        "success": success,
        "dense_linearization_materialized": dense_linearization_materialized,
        "dense_linearization_kind": (
            "post_hoc" if dense_linearization_materialized else None
        ),
        **dense_report,
    }


def levenberg_marquardt_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=None,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe Levenberg-Marquardt solver for least-squares residuals.

    ``ftol`` and ``xtol`` feed the matrix-free MINPACK-style ``info`` subset.
    Explicit ``gtol`` values replace the legacy ``tol`` gradient gate with a
    matrix-free infinity-norm gradient gate. QR-scaled ``gtol`` ``info`` codes
    remain reserved for a future pivoted-QR MINPACK lane.
    """
    runner = _make_traceable_levenberg_marquardt_runner(
        residual_fn,
        int(maxiter),
        float(tol),
        float(ftol),
        float(xtol),
        None if gtol is None else float(gtol),
        bool(materialize_dense_linearization),
        max_dense_linearization_bytes,
        callback,
        progress_callback,
    )
    return runner(x0, _normalize_solver_args(args))


def _qr_lm_dense_state(flat_residual_fn, flat_x):
    residual = flat_residual_fn(flat_x)

    def jvp_fn(x, v):
        return jax.jvp(flat_residual_fn, (x,), (v,))[1]

    jacobian = _materialize_dense_jacobian(jvp_fn, flat_x)
    gradient, hessian = _least_squares_linearization_from_jacobian(
        residual,
        jacobian,
    )
    return {
        "residual": residual,
        "jacobian": jacobian,
        "gradient": gradient,
        "hessian": hessian,
        "cost": _least_squares_cost(residual),
        "grad_norm_inf": _tree_inf_norm(gradient),
    }


def _qr_scaled_gradient_norm(residual, jacobian):
    residual = jnp.ravel(jnp.asarray(residual))
    jacobian = jnp.asarray(jacobian)
    dtype = residual.dtype
    residual_norm = jnp.linalg.norm(residual)
    column_norms = jnp.linalg.norm(jacobian, axis=0)
    numerator = jnp.abs(jacobian.T @ residual)
    denominator = column_norms * residual_norm
    cosines = jnp.where(
        denominator > _device_scalar(0.0, dtype=dtype),
        numerator / denominator,
        _device_scalar(0.0, dtype=dtype),
    )
    return jnp.max(cosines)


def _qr_lm_info(
    *,
    actual_reduction,
    predicted_reduction,
    cost,
    delta,
    x_norm,
    nit,
    maxiter,
    ftol,
    xtol,
    gtol,
    epsmch,
    qr_gnorm,
):
    info = _matrix_free_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=cost,
        delta=delta,
        x_norm=x_norm,
        nit=nit,
        maxiter=maxiter,
        ftol=ftol,
        xtol=xtol,
        epsmch=epsmch,
    )
    info = jnp.where(
        (info == 0) & (qr_gnorm <= gtol),
        jnp.asarray(4, dtype=jnp.int32),
        info,
    )
    info = jnp.where(
        (info == 0) & (qr_gnorm <= epsmch),
        jnp.asarray(8, dtype=jnp.int32),
        info,
    )
    return info


def _qr_lm_step(jacobian, residual, damping):
    jacobian = jnp.asarray(jacobian)
    residual = jnp.ravel(jnp.asarray(residual))
    cols = jacobian.shape[1]
    dtype = jacobian.dtype
    damping_sqrt = jnp.sqrt(jnp.asarray(damping, dtype=dtype))
    augmented_jacobian = jnp.concatenate(
        (jacobian, damping_sqrt * jnp.eye(cols, dtype=dtype)),
        axis=0,
    )
    augmented_rhs = jnp.concatenate(
        (-residual, jnp.zeros(cols, dtype=dtype)),
        axis=0,
    )
    q_matrix, r_matrix, pivots = jsp_linalg.qr(
        augmented_jacobian,
        pivoting=True,
        mode="economic",
    )
    pivoted_step = jsp_linalg.solve_triangular(
        r_matrix,
        q_matrix.T @ augmented_rhs,
        lower=False,
    )
    return jnp.zeros_like(pivoted_step).at[pivots].set(pivoted_step)


def _qr_lm_iteration(
    flat_residual_fn,
    state,
    *,
    gradient_tol,
    ftol,
    xtol,
    gtol,
    maxiter,
):
    dtype = state["x"].dtype
    defaults = _lm_defaults(dtype)
    damping = _clip_lm_damping(state["damping"], dtype=dtype)
    step = _qr_lm_step(state["jacobian"], state["residual"], damping)
    x_candidate = state["x"] + step
    candidate = _qr_lm_dense_state(flat_residual_fn, x_candidate)

    predicted_residual = state["residual"] + state["jacobian"] @ step
    predicted_reduction = state["cost"] - _least_squares_cost(predicted_residual)
    actual_reduction = state["cost"] - candidate["cost"]
    ratio = actual_reduction / jnp.maximum(
        predicted_reduction,
        defaults["predicted_floor"],
    )
    finite_candidate = (
        jnp.all(jnp.isfinite(x_candidate))
        & jnp.all(jnp.isfinite(step))
        & jnp.all(jnp.isfinite(candidate["residual"]))
        & jnp.all(jnp.isfinite(candidate["jacobian"]))
        & jnp.all(jnp.isfinite(candidate["gradient"]))
        & jnp.all(jnp.isfinite(candidate["hessian"]))
        & jnp.isfinite(candidate["cost"])
        & jnp.isfinite(predicted_reduction)
    )
    accepted = finite_candidate & (ratio >= defaults["accept_threshold"])
    step_norm = jnp.linalg.norm(step)
    delta_after_step = _lm_delta_after_step(
        state["delta"],
        step_norm,
        jnp.asarray(ratio, dtype=state["delta"].dtype),
        jnp.asarray(actual_reduction, dtype=state["delta"].dtype),
        defaults=defaults,
    )
    damping_after_accept = lax.cond(
        ratio > defaults["ratio_high"],
        lambda _: damping * defaults["decrease_factor"],
        lambda _: lax.cond(
            ratio < defaults["ratio_low"],
            lambda __: damping * defaults["increase_factor"],
            lambda __: damping,
            operand=None,
        ),
        operand=None,
    )
    next_damping = lax.cond(
        accepted,
        lambda _: _clip_lm_damping(damping_after_accept, dtype=dtype),
        lambda _: _clip_lm_damping(
            damping * defaults["increase_factor"],
            dtype=dtype,
        ),
        operand=None,
    )
    x_next = lax.select(accepted, x_candidate, state["x"])
    residual_next = lax.select(accepted, candidate["residual"], state["residual"])
    jacobian_next = lax.select(accepted, candidate["jacobian"], state["jacobian"])
    gradient_next = lax.select(accepted, candidate["gradient"], state["gradient"])
    hessian_next = lax.select(accepted, candidate["hessian"], state["hessian"])
    cost_next = lax.select(accepted, candidate["cost"], state["cost"])
    grad_norm_next = lax.select(
        accepted,
        candidate["grad_norm_inf"],
        state["grad_norm_inf"],
    )
    x_norm = jnp.linalg.norm(x_next)
    next_nit = state["nit"] + 1
    qr_gnorm = _qr_scaled_gradient_norm(residual_next, jacobian_next)
    epsmch = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    info_candidate = _qr_lm_info(
        actual_reduction=actual_reduction,
        predicted_reduction=predicted_reduction,
        cost=state["cost"],
        delta=delta_after_step,
        x_norm=x_norm,
        nit=next_nit,
        maxiter=jnp.asarray(maxiter, dtype=jnp.int32),
        ftol=jnp.asarray(ftol, dtype=dtype),
        xtol=jnp.asarray(xtol, dtype=dtype),
        gtol=jnp.asarray(gtol, dtype=dtype),
        epsmch=epsmch,
        qr_gnorm=qr_gnorm,
    )
    info_next = lax.select(
        finite_candidate,
        info_candidate,
        jnp.asarray(0, dtype=jnp.int32),
    )
    legacy_success = grad_norm_next <= gradient_tol
    info_success = (
        (info_next == 1) | (info_next == 2) | (info_next == 3) | (info_next == 4)
    )

    return {
        "x": x_next,
        "residual": residual_next,
        "jacobian": jacobian_next,
        "gradient": gradient_next,
        "hessian": hessian_next,
        "cost": cost_next,
        "grad_norm_inf": grad_norm_next,
        "damping": next_damping,
        "delta": delta_after_step,
        "nit": next_nit,
        "status": lax.select(
            finite_candidate,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
        ),
        "info": info_next,
        "accepted": accepted,
        "success": finite_candidate & (legacy_success | info_success),
    }


def levenberg_marquardt_minpack_traceable(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe QR Levenberg-Marquardt solver for least-squares residuals.

    This opt-in lane materializes the dense Jacobian each iteration and solves
    the Marquardt augmented least-squares system with column-pivoted QR. It is
    MINPACK-style and tolerance-equivalent; it does not claim MINPACK packed-QR
    byte identity.
    """
    x = jax.tree_util.tree_map(jnp.asarray, x0)
    flat_x0, unravel = ravel_pytree(x)
    normalized_args = _normalize_solver_args(args)
    dtype = flat_x0.dtype
    gradient_tol = _lm_gradient_tol(tol, gtol, dtype=dtype)
    gtol_value = _device_scalar(gtol, dtype=dtype)

    def residual_eval(flat_x):
        return jnp.ravel(jnp.asarray(residual_fn(unravel(flat_x), *normalized_args)))

    residual0 = residual_eval(flat_x0)
    linearization_rows = int(np.asarray(jnp.asarray(residual0).size))
    linearization_cols = int(np.asarray(jnp.asarray(flat_x0).size))
    dense_linearization_within_budget, dense_report = (
        _least_squares_dense_linearization_policy(
            linearization_rows,
            linearization_cols,
            dtype,
            max_dense_linearization_bytes,
        )
    )
    if not dense_linearization_within_budget:
        raise MemoryError(
            _least_squares_required_dense_linearization_message(
                linearization_rows,
                linearization_cols,
                dtype,
                max_dense_linearization_bytes,
            )
        )

    def run_solver(flat_x_init):
        initial = _qr_lm_dense_state(residual_eval, flat_x_init)
        initial_qr_gnorm = _qr_scaled_gradient_norm(
            initial["residual"],
            initial["jacobian"],
        )
        initial_info = jnp.where(
            initial_qr_gnorm <= gtol_value,
            jnp.asarray(4, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        state0 = {
            "x": flat_x_init,
            "residual": initial["residual"],
            "jacobian": initial["jacobian"],
            "gradient": initial["gradient"],
            "hessian": initial["hessian"],
            "cost": initial["cost"],
            "grad_norm_inf": initial["grad_norm_inf"],
            "damping": _lm_defaults(dtype)["initial_damping"],
            "delta": _lm_defaults(dtype)["initial_delta_factor"]
            * jnp.maximum(
                jnp.linalg.norm(flat_x_init),
                _device_scalar(1.0, dtype=dtype),
            ),
            "nit": jnp.asarray(0, dtype=jnp.int32),
            "status": jnp.asarray(0, dtype=jnp.int32),
            "info": initial_info,
            "accepted": jnp.asarray(False),
            "success": (initial["grad_norm_inf"] <= gradient_tol) | (initial_info == 4),
        }

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (~state["success"])
                & (state["status"] != 2)
                & (state["info"] == 0)
            )

        def body_fun(state):
            next_state = _qr_lm_iteration(
                residual_eval,
                state,
                gradient_tol=gradient_tol,
                ftol=_device_scalar(ftol, dtype=dtype),
                xtol=_device_scalar(xtol, dtype=dtype),
                gtol=gtol_value,
                maxiter=maxiter,
            )
            if callback is not None:
                lax.cond(
                    next_state["accepted"],
                    lambda _: jax.debug.callback(
                        lambda flat_x: callback(
                            _hostify_optimizer_tree(unravel(flat_x))
                        ),
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

        return lax.while_loop(cond_fun, body_fun, state0)

    state = jax.jit(run_solver)(flat_x0)
    return {
        "x": unravel(state["x"]),
        "residual": state["residual"],
        "residual_jacobian": state["jacobian"],
        "fun": state["cost"],
        "grad": unravel(state["gradient"]),
        "hessian": state["hessian"],
        "damping": state["damping"],
        "nit": state["nit"],
        "status": state["status"],
        "info": state["info"],
        "success": state["success"],
        "dense_linearization_materialized": True,
        "dense_linearization_kind": "in_loop",
        **dense_report,
    }


def jax_least_squares_optimistix(
    residual_fn,
    x0,
    *,
    maxiter=1500,
    tol=1e-10,
    ftol=_OPTIMISTIX_LM_DEFAULT_FTOL,
    xtol=_OPTIMISTIX_LM_DEFAULT_XTOL,
    gtol=_OPTIMISTIX_LM_DEFAULT_GTOL,
    materialize_dense_linearization=True,
    max_dense_linearization_bytes=None,
    callback=None,
    progress_callback=None,
    args=(),
):
    """Optional Optimistix/Lineax LSMR least-squares target lane."""
    _require_optimistix_lm_contract_options(
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        callback=callback,
        progress_callback=progress_callback,
    )

    try:
        import lineax
        import optimistix as optx
    except ImportError as exc:
        raise ImportError(
            "method='optimistix-lm-ondevice' requires the optional "
            "Optimistix/Lineax dependencies. Install with "
            "simsopt[JAX_OPTIMISTIX] on Python 3.11+."
        ) from exc

    x = jax.tree_util.tree_map(jnp.asarray, x0)
    normalized_args = _normalize_solver_args(args)
    dtype = _require_tree_first_leaf(
        x,
        detail="Least-squares initial state must contain at least one leaf.",
    ).dtype
    tol_value = float(tol)

    def residual_eval(x_current):
        return jnp.ravel(jnp.asarray(residual_fn(x_current, *normalized_args)))

    def optx_residual(x_current, fn_args):
        return jnp.ravel(jnp.asarray(residual_fn(x_current, *fn_args)))

    solver = optx.LevenbergMarquardt(
        rtol=tol_value,
        atol=tol_value,
        linear_solver=lineax.LSMR(rtol=tol_value, atol=tol_value),
    )
    solution = optx.least_squares(
        optx_residual,
        solver,
        x,
        args=normalized_args,
        max_steps=int(maxiter),
        throw=False,
    )

    residual, cost, grad, grad_norm_inf, _ = _least_squares_gradient_state(
        residual_eval,
        solution.value,
    )
    linearization_rows = int(np.asarray(jnp.asarray(residual).size))
    linearization_cols = sum(
        int(np.asarray(jnp.asarray(leaf).size))
        for leaf in jax.tree_util.tree_leaves(solution.value)
    )
    residual_jacobian = None
    hessian = None
    dense_linearization_materialized = bool(materialize_dense_linearization)
    if dense_linearization_materialized:
        dense_linearization_materialized, dense_report = (
            _least_squares_dense_linearization_policy(
                linearization_rows,
                linearization_cols,
                dtype,
                max_dense_linearization_bytes,
            )
        )
        if dense_linearization_materialized:
            residual, residual_jacobian, _flat_grad, hessian = (
                _materialize_dense_least_squares_linearization(
                    residual_eval,
                    solution.value,
                )
            )
    else:
        dense_report = _least_squares_dense_linearization_report(
            linearization_rows,
            linearization_cols,
            dtype,
            max_dense_linearization_bytes,
        )
        dense_report["failure_category"] = None
        dense_report["failure_stage"] = None
        dense_report["message"] = None

    finite = (
        _tree_all_finite(solution.value)
        & jnp.all(jnp.isfinite(residual))
        & _tree_all_finite(grad)
        & jnp.isfinite(cost)
    )
    if hessian is not None:
        finite = finite & jnp.all(jnp.isfinite(hessian))
    solution_success = jnp.asarray(solution.result == optx.RESULTS.successful)
    max_steps_reached = jnp.asarray(
        (solution.result == optx.RESULTS.nonlinear_max_steps_reached)
        | (solution.result == optx.RESULTS.max_steps_reached)
    )
    status = jnp.where(
        finite,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
    )
    info = jnp.where(
        max_steps_reached,
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return {
        "x": solution.value,
        "residual": residual,
        "residual_jacobian": residual_jacobian,
        "fun": cost,
        "grad": grad,
        "grad_norm_inf": grad_norm_inf,
        "hessian": hessian,
        "damping": None,
        "nit": jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
        "status": status,
        "info": info,
        "success": solution_success & finite,
        "dense_linearization_materialized": dense_linearization_materialized,
        "dense_linearization_kind": (
            "post_hoc" if dense_linearization_materialized else None
        ),
        "optimistix_result": str(solution.result),
        "optimistix_result_message": optx.RESULTS[solution.result],
        **dense_report,
    }


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


def _materialize_dense_hessian(hvp_fn, x, *, symmetrize=True):
    dense = _materialize_dense_linear_operator(hvp_fn, x)
    if not bool(symmetrize):
        return dense
    upper = jnp.triu(dense)
    return upper + jnp.triu(dense, 1).T


def _materialize_dense_jacobian(jvp_fn, x):
    return _materialize_dense_linear_operator(jvp_fn, x)


def _dense_operator_nbytes(rows, cols, dtype):
    return int(rows) * int(cols) * np.dtype(dtype).itemsize


def _dense_operator_exceeds_bytes_limit(rows, cols, dtype, max_dense_bytes):
    if max_dense_bytes is None:
        return False
    return _dense_operator_nbytes(rows, cols, dtype) > int(max_dense_bytes)


def _dense_square_operator_report(name, size, dtype, max_dense_bytes):
    return {
        f"dense_{name}_shape": (int(size), int(size)),
        f"dense_{name}_bytes": _dense_operator_nbytes(size, size, dtype),
        f"max_dense_{name}_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _dense_square_operator_message(
    *,
    solver_name,
    artifact_name,
    size,
    dtype,
    max_dense_bytes,
):
    required_bytes = _dense_operator_nbytes(size, size, dtype)
    return (
        f"{solver_name} skipped dense {artifact_name} materialization because "
        f"the final {int(size)}x{int(size)} matrix in dtype {np.dtype(dtype)} "
        f"would require {required_bytes} bytes, exceeding "
        f"max_dense_{artifact_name}_bytes={int(max_dense_bytes)}."
    )


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
    stab_value = _optimizer_scalar(stab, dtype=H.dtype)
    return H + stab_value * jnp.eye(H.shape[0], dtype=H.dtype)


def _solve_dense_newton_step(H, grad, *, refine):
    H_host = np.asarray(H, dtype=np.float64)
    grad_host = np.asarray(grad, dtype=np.float64)
    dx = np.linalg.solve(H_host, grad_host)
    if refine:
        dx = dx + np.linalg.solve(H_host, grad_host - H_host @ dx)
    return jnp.asarray(dx, dtype=jnp.asarray(grad).dtype)


def _factor_dense_hessian(H, *, optimizer_backend):
    """Factor a dense LS Hessian once and return packed ``(lu, piv)``.

    Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §5.3
    (Phase 2 adjoint factor-once hybrid). The resulting factors are reused
    for both forward and adjoint solves so the bytes are bit-identical by
    construction.

    The ``optimizer_backend == "scipy"`` branch routes through host LAPACK
    ``dgetrf`` via ``scipy.linalg.lu_factor`` so the LS reference lane keeps
    matching CPU pivot tie-breaks. All other backends call
    ``jax.scipy.linalg.lu_factor`` on ``H``'s device, which dispatches to
    LAPACK on CPU and cuSOLVER ``getrf`` on CUDA. Both APIs use the same
    0-indexed packed pivot semantics, so the returned ``(lu, piv)`` is a
    drop-in to ``jax.scipy.linalg.lu_solve``.
    """
    if H is None:
        return None
    if optimizer_backend == "scipy":
        H_host = np.asarray(H, dtype=np.float64)
        lu_host, piv_host = scipy.linalg.lu_factor(H_host)
        lu = jnp.asarray(lu_host, dtype=H.dtype)
        piv = jnp.asarray(piv_host, dtype=jnp.int32)
        return lu, piv
    return jsp_linalg.lu_factor(H)


def _lu_solve_dense_hessian(lu_piv, rhs, *, transpose):
    """Solve a dense LS Hessian system from packed ``(lu, piv)`` factors.

    Routes through ``jax.scipy.linalg.lu_solve`` with ``trans=1`` for the
    transpose path so adjoint and forward solves consume the same packed
    factor bytes. Pivot reconstruction stays inside the LAPACK/cuSOLVER
    contract; no manual ``_piv_from(P)`` rebuilding happens at the call
    site.
    """
    lu, piv = lu_piv
    trans = 1 if transpose else 0
    return jsp_linalg.lu_solve((lu, piv), rhs, trans=trans)


@jax.jit
def _plu_from_lu_piv(lu_piv):
    """Derive ``(P, L, U)`` matrices from packed ``(lu, piv)`` factors.

    Used for backward-compatible reporting under the
    ``"dense-plu-shared"`` factorization backend: the ``res["PLU"]`` slot
    keeps surfacing the public triple while the runtime forward and
    adjoint solves consume the same ``(lu, piv)`` factor bytes. The
    permutation array is built with ``lax.fori_loop`` so the helper is
    JIT-traceable; the ``jax.jit`` wrapper hoists the static-shape
    ``jnp.eye`` / ``jnp.zeros`` constructors inside the trace so callers
    in strict transfer-guard contexts do not pay a host roundtrip per
    invocation.
    """
    lu, piv = lu_piv
    n = lu.shape[0]
    eye = jnp.eye(n, dtype=lu.dtype)
    L = jnp.tril(lu, k=-1) + eye
    U = jnp.triu(lu)

    def body(i, perm):
        a = perm[i]
        b = perm[piv[i]]
        perm = perm.at[i].set(b)
        perm = perm.at[piv[i]].set(a)
        return perm

    perm = lax.fori_loop(0, n, body, jnp.arange(n, dtype=jnp.int32))
    columns = jnp.arange(n, dtype=jnp.int32)
    P = (
        jnp.zeros((n, n), dtype=lu.dtype)
        .at[perm, columns]
        .set(jnp.asarray(1.0, dtype=lu.dtype))
    )
    return P, L, U


def _least_squares_dense_hessian_report(size, dtype, max_dense_bytes):
    return _dense_square_operator_report(
        "hessian",
        size,
        dtype,
        max_dense_bytes,
    )


def _least_squares_dense_hessian_message(size, dtype, max_dense_bytes):
    return _dense_square_operator_message(
        solver_name="Newton polish",
        artifact_name="hessian",
        size=size,
        dtype=dtype,
        max_dense_bytes=max_dense_bytes,
    )


def _least_squares_dense_hessian_policy(size, dtype, max_dense_bytes):
    report = _least_squares_dense_hessian_report(size, dtype, max_dense_bytes)
    materialize_hessian = not _dense_operator_exceeds_bytes_limit(
        size,
        size,
        dtype,
        max_dense_bytes,
    )
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_hessian:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_hessian_finalization"
        report["message"] = _least_squares_dense_hessian_message(
            size,
            dtype,
            max_dense_bytes,
        )
    return materialize_hessian, report


def _resolve_dense_hessian_materialization(
    requested,
    size,
    dtype,
    max_dense_bytes,
):
    if not requested:
        report = _least_squares_dense_hessian_report(size, dtype, max_dense_bytes)
        report["failure_category"] = None
        report["failure_stage"] = None
        report["message"] = None
        return False, report
    return _least_squares_dense_hessian_policy(size, dtype, max_dense_bytes)


def _least_squares_dense_linearization_report(rows, cols, dtype, max_dense_bytes):
    jacobian_bytes = _dense_operator_nbytes(rows, cols, dtype)
    hessian_bytes = _dense_operator_nbytes(cols, cols, dtype)
    return {
        "dense_residual_jacobian_shape": (int(rows), int(cols)),
        "dense_residual_jacobian_bytes": jacobian_bytes,
        "dense_hessian_shape": (int(cols), int(cols)),
        "dense_hessian_bytes": hessian_bytes,
        "dense_linearization_bytes": jacobian_bytes + hessian_bytes,
        "max_dense_linearization_bytes": (
            None if max_dense_bytes is None else int(max_dense_bytes)
        ),
    }


def _least_squares_dense_linearization_message(rows, cols, dtype, max_dense_bytes):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    return (
        "Levenberg-Marquardt skipped dense linearization materialization because "
        f"the final residual Jacobian/Hessian compatibility artifacts would "
        f"require {report['dense_linearization_bytes']} bytes in dtype "
        f"{np.dtype(dtype)}, exceeding "
        f"max_dense_linearization_bytes={int(max_dense_bytes)}."
    )


def _least_squares_required_dense_linearization_message(
    rows,
    cols,
    dtype,
    max_dense_bytes,
):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    return (
        "Levenberg-Marquardt dense QR solve requires residual Jacobian/Hessian "
        f"artifacts totaling {report['dense_linearization_bytes']} bytes in "
        f"dtype {np.dtype(dtype)}, exceeding "
        f"max_dense_linearization_bytes={int(max_dense_bytes)}."
    )


def _least_squares_dense_linearization_policy(rows, cols, dtype, max_dense_bytes):
    report = _least_squares_dense_linearization_report(
        rows,
        cols,
        dtype,
        max_dense_bytes,
    )
    materialize_linearization = max_dense_bytes is None or report[
        "dense_linearization_bytes"
    ] <= int(max_dense_bytes)
    report["failure_category"] = None
    report["failure_stage"] = None
    report["message"] = None
    if not materialize_linearization:
        report["failure_category"] = "scaling_limit"
        report["failure_stage"] = "dense_linearization_finalization"
        report["message"] = _least_squares_dense_linearization_message(
            rows,
            cols,
            dtype,
            max_dense_bytes,
        )
    return materialize_linearization, report


def _newton_step_finite(x_next, grad_next):
    return jnp.all(jnp.isfinite(x_next)) & jnp.all(jnp.isfinite(grad_next))


def _newton_candidate_status(x_next, val_next, grad_next):
    candidate_norm = jnp.linalg.norm(grad_next)
    accepted = (
        _newton_step_finite(x_next, grad_next)
        & jnp.isfinite(val_next)
        & jnp.isfinite(candidate_norm)
    )
    return accepted, candidate_norm


def _newton_backtracking_continue(state):
    return (state["iteration"] < _NEWTON_BACKTRACKING_MAX_STEPS) & (~state["accepted"])


def _backtracking_value_grad_step(
    val_and_grad_fn,
    x,
    dx,
    current_val,
    current_grad,
    current_norm,
):
    dtype = jnp.asarray(x).dtype
    one = _device_scalar(1.0, dtype=dtype)
    half = _device_scalar(0.5, dtype=dtype)
    state0 = {
        "iteration": jnp.asarray(0, dtype=jnp.int32),
        "alpha": one,
        "x": x,
        "val": current_val,
        "grad": current_grad,
        "norm": current_norm,
        "accepted": jnp.asarray(False),
    }

    def body_fun(state):
        candidate_x = x - state["alpha"] * dx
        candidate_val, candidate_grad = val_and_grad_fn(candidate_x)
        candidate_accepted, candidate_norm = _newton_candidate_status(
            candidate_x,
            candidate_val,
            candidate_grad,
        )
        candidate_accepted = candidate_accepted & (candidate_norm <= current_norm)
        return {
            "iteration": state["iteration"] + 1,
            "alpha": state["alpha"] * half,
            "x": lax.select(candidate_accepted, candidate_x, state["x"]),
            "val": lax.select(candidate_accepted, candidate_val, state["val"]),
            "grad": lax.select(candidate_accepted, candidate_grad, state["grad"]),
            "norm": lax.select(candidate_accepted, candidate_norm, state["norm"]),
            "accepted": candidate_accepted,
        }

    return lax.while_loop(_newton_backtracking_continue, body_fun, state0)


def _backtracking_residual_step(residual_eval, x, dx, residual, current_norm):
    dtype = jnp.asarray(x).dtype
    one = _device_scalar(1.0, dtype=dtype)
    half = _device_scalar(0.5, dtype=dtype)
    state0 = {
        "iteration": jnp.asarray(0, dtype=jnp.int32),
        "alpha": one,
        "x": x,
        "residual": residual,
        "norm": current_norm,
        "accepted": jnp.asarray(False),
    }

    def body_fun(state):
        candidate_x = x - state["alpha"] * dx
        candidate_residual = residual_eval(candidate_x)
        candidate_norm = jnp.linalg.norm(candidate_residual)
        candidate_accepted = (
            jnp.all(jnp.isfinite(candidate_x))
            & jnp.all(jnp.isfinite(candidate_residual))
            & jnp.isfinite(candidate_norm)
            & (candidate_norm <= current_norm)
        )
        return {
            "iteration": state["iteration"] + 1,
            "alpha": state["alpha"] * half,
            "x": lax.select(candidate_accepted, candidate_x, state["x"]),
            "residual": lax.select(
                candidate_accepted,
                candidate_residual,
                state["residual"],
            ),
            "norm": lax.select(candidate_accepted, candidate_norm, state["norm"]),
            "accepted": candidate_accepted,
        }

    return lax.while_loop(_newton_backtracking_continue, body_fun, state0)


def _gmres_iteration_limits(n):
    restart = max(5, min(n, 64))
    maxiter = 10
    return restart, maxiter


def _run_operator_gmres(matvec, rhs, *, tol):
    n = rhs.shape[0]
    restart, maxiter = _gmres_iteration_limits(n)
    # JAX's gmres implementation currently lowers a few scalar literals through
    # host-to-device conversions even when the caller provides fully device-
    # resident operands. Keep the allowance scoped to the library call so the
    # surrounding operator path remains strict-transfer clean.
    with jax.transfer_guard("allow"):
        return gmres(
            matvec,
            rhs,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            # JAX documents the incremental method as numerically stabler than the
            # default batched variant, which matters more than lower GPU overhead
            # on the checked operator-only runtime path.
            solve_method="incremental",
        )


def _gmres_solve_newton_system(hvp_fn, x, rhs, *, stab, tol):
    stab_value = _optimizer_scalar(stab, dtype=rhs.dtype)

    def matvec(v):
        return hvp_fn(x, v) + stab_value * v

    dx, _ = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(dx)
    return dx, residual, matvec


def _gmres_solve_exact_newton_system(jvp_fn, x, rhs, *, tol):
    def matvec(v):
        return jvp_fn(x, v)

    dx, _ = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(dx)
    return dx, residual, matvec


def _gmres_solve_array_system(matvec, rhs, *, tol):
    solution, _ = _run_operator_gmres(matvec, rhs, tol=tol)
    residual = rhs - matvec(solution)
    return solution, residual


def _linear_solve_finite(solution, residual):
    return jnp.all(jnp.isfinite(solution)) & jnp.all(jnp.isfinite(residual))


def _linear_solve_residual_tolerance(rhs, tol):
    dtype = rhs.dtype
    rhs_norm = jnp.linalg.norm(rhs)
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    one = _device_scalar(1.0, dtype=dtype)
    ten = _device_scalar(10.0, dtype=dtype)
    minimum = _device_scalar(1e-12, dtype=dtype)
    scale = jnp.maximum(rhs_norm, one)
    return jnp.maximum(
        minimum,
        ten * tol_value * scale,
    )


def _relative_residual_norm(residual, rhs, *, ord=None):
    """Return ``||residual|| / max(||rhs||, 1)`` when ``||rhs||`` is not
    representably nonzero (denormals included), else ``||residual|| /
    ||rhs||``. The unit fallback prevents the denormal floor from
    inflating residual_rel by ~10^308 when the RHS is the zero vector
    (a legitimate degenerate adjoint state) which would otherwise force
    the forward-error gate to spurious failure.
    """
    dtype = rhs.dtype
    residual_norm = jnp.linalg.norm(residual, ord=ord)
    rhs_norm = jnp.linalg.norm(rhs, ord=ord)
    tiny = _device_scalar(jnp.finfo(dtype).tiny, dtype=dtype)
    one = _device_scalar(1.0, dtype=dtype)
    safe_norm = jnp.where(rhs_norm > tiny, rhs_norm, one)
    return residual_norm / safe_norm


def _relative_residual_1_norm(residual, rhs):
    return _relative_residual_norm(residual, rhs, ord=1)


def _forward_error_bound(residual_rel, condition_estimate):
    dtype = residual_rel.dtype
    one = _device_scalar(1.0, dtype=dtype)
    inf_value = _device_scalar(jnp.inf, dtype=dtype)
    scaled = condition_estimate * residual_rel
    denominator = one - scaled
    return jnp.where(
        denominator > _device_scalar(0.0, dtype=dtype),
        scaled / denominator,
        inf_value,
    )


def _forward_error_success(residual_rel, condition_estimate, *, tol):
    dtype = residual_rel.dtype
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    floor = jnp.sqrt(_device_scalar(jnp.finfo(dtype).eps, dtype=dtype))
    gate = jnp.maximum(floor, _device_scalar(10.0, dtype=dtype) * tol_value)
    ferr = _forward_error_bound(residual_rel, condition_estimate)
    return jnp.isfinite(ferr) & (ferr <= gate)


def _eisenstat_walker_choice2_tolerance(norm, previous_norm, *, tol):
    """Return the Eisenstat-Walker Choice-2 relative linear-solve tolerance.

    Eisenstat & Walker, "Choosing the Forcing Terms in an Inexact Newton
    Method," SIAM J. Sci. Comput. 17(1):16-32 (1996), eq. (2.6) with
    γ=0.9, α=2. The returned value is the **relative** linear residual
    tolerance (`||A·dx + r_k|| ≤ η · ||r_k||`) consumed directly as
    `tol=` by `jax.scipy.sparse.linalg.gmres`, which interprets `tol` as
    relative to `||rhs||`. A fixed strict cap from the legacy contract
    bounds the value from above so the linear solve never undercuts the
    Newton convergence target.
    """
    dtype = norm.dtype
    tol_value = _optimizer_scalar(tol, dtype=dtype)
    strict_cap = jnp.minimum(
        _device_scalar(1e-10, dtype=dtype),
        jnp.maximum(
            tol_value * _device_scalar(0.1, dtype=dtype),
            _device_scalar(1e-14, dtype=dtype),
        ),
    )
    gamma = _device_scalar(_EISENSTAT_WALKER_GAMMA, dtype=dtype)
    eta_min = _device_scalar(_EISENSTAT_WALKER_MIN_ETA, dtype=dtype)
    eta_max = _device_scalar(_EISENSTAT_WALKER_MAX_ETA, dtype=dtype)
    denominator = jnp.maximum(
        previous_norm,
        _device_scalar(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    ratio = norm / denominator
    eta = gamma * (ratio * ratio)
    eta = jnp.clip(eta, eta_min, eta_max)
    return jnp.maximum(
        _device_scalar(1e-14, dtype=dtype),
        jnp.minimum(strict_cap, eta),
    )


def _matrix_one_norm(matrix):
    return jnp.max(jnp.sum(jnp.abs(matrix), axis=0))


def _hager_higham_inverse_1_norm_estimate(
    solve,
    transpose_solve,
    *,
    size,
    dtype,
    iterations=_HAGER_HIGHAM_CONDITION_ITERATIONS,
):
    one = _device_scalar(1.0, dtype=dtype)
    zero = _device_scalar(0.0, dtype=dtype)
    indices = jnp.arange(size)
    x0 = jnp.full((size,), one / _device_scalar(size, dtype=dtype), dtype=dtype)

    def unit_vector(index):
        return jnp.where(indices == index, one, zero)

    inf_value = _device_scalar(jnp.inf, dtype=dtype)

    def body_fun(_iteration, state):
        x, best_estimate = state
        y = solve(x)
        estimate = jnp.sum(jnp.abs(y))
        signs = jnp.where(y >= zero, one, -one)
        z = transpose_solve(signs)
        next_index = jnp.argmax(jnp.abs(z))
        next_x = unit_vector(next_index)
        finite = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(z))
        next_estimate = jnp.maximum(best_estimate, estimate)
        return next_x, jnp.where(finite, next_estimate, inf_value)

    # ``lax.fori_loop`` lowers the Python integer bounds through a weakly
    # typed host-to-device conversion that strict transfer-guard contexts
    # flag as a violation. Mirror the ``_run_operator_gmres`` allowance:
    # scope the relaxation to the library call so the surrounding solve
    # path stays strict-transfer clean.
    with jax.transfer_guard("allow"):
        _, estimate = lax.fori_loop(0, int(iterations), body_fun, (x0, zero))
    return estimate


def _dense_matrix_condition_estimate(matrix, *, lu_piv=None):
    """Return a JAX-native Hager-Higham 1-norm condition estimate.

    The Hager-Higham iteration evaluates ``A⁻¹`` and ``A⁻ᵀ`` repeatedly,
    so the inner solves consume cached ``(lu, piv)`` factors via
    ``jsp_linalg.lu_solve``. When ``lu_piv`` is supplied (e.g., the
    Phase-2 5-tuple ``(P, L, U, lu, piv)`` ``linear_solve_factors``
    snapshot) no factorization runs at all; otherwise the helper
    factorizes ``matrix`` once and shares those bytes across every
    inner solve. The naïve ``jnp.linalg.solve`` form re-factorized for
    every call, costing 10 × O(n³) per estimate instead of the present
    O(n³) + 10 × O(n²).
    """
    matrix = jnp.asarray(matrix)
    size = int(matrix.shape[0])

    if lu_piv is None:
        lu_piv = jsp_linalg.lu_factor(matrix)
    lu, piv = lu_piv

    def solve(rhs):
        return jsp_linalg.lu_solve((lu, piv), rhs, trans=0)

    def transpose_solve(rhs):
        return jsp_linalg.lu_solve((lu, piv), rhs, trans=1)

    matrix_norm = _matrix_one_norm(matrix)
    inverse_norm = _hager_higham_inverse_1_norm_estimate(
        solve,
        transpose_solve,
        size=size,
        dtype=matrix.dtype,
    )
    return matrix_norm * inverse_norm


def _dense_matrix_solve_forward_error_success(matrix, solution, rhs, *, tol):
    residual = rhs - matrix @ solution
    residual_rel = _relative_residual_1_norm(residual, rhs)
    condition_estimate = _dense_matrix_condition_estimate(matrix)
    return _forward_error_success(residual_rel, condition_estimate, tol=tol)


def _solve_square_vector_system_operator_only(matvec, rhs, *, tol):
    """Solve one square linear system with operator-only GMRES refinement."""
    solution, residual = _gmres_solve_array_system(matvec, rhs, tol=tol)
    residual_norm = jnp.linalg.norm(residual)
    residual_tol = _linear_solve_residual_tolerance(rhs, tol)
    solve_finite = _linear_solve_finite(solution, residual) & jnp.isfinite(
        residual_norm
    )

    def refine(_):
        correction, correction_residual = _gmres_solve_array_system(
            matvec,
            residual,
            tol=tol,
        )
        correction_finite = _linear_solve_finite(correction, correction_residual)
        refined_solution = lax.cond(
            correction_finite,
            lambda _: solution + correction,
            lambda _: solution,
            operand=None,
        )
        refined_residual = rhs - matvec(refined_solution)
        return refined_solution, refined_residual

    solution, residual = lax.cond(
        solve_finite & (residual_norm > residual_tol),
        refine,
        lambda _: (solution, residual),
        operand=None,
    )
    residual_norm = jnp.linalg.norm(residual)
    success = (
        _linear_solve_finite(solution, residual)
        & jnp.isfinite(residual_norm)
        & (residual_norm <= residual_tol)
    )
    return solution, success


def _apply_column_batched_operator(matvec, rhs):
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 1:
        return matvec(rhs)
    return jax.vmap(matvec, in_axes=1, out_axes=1)(rhs)


def _solve_square_array_system_operator_only(matvec, rhs, *, tol):
    """Solve vector or column-batched square systems with operator-only GMRES."""
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 1:
        return _solve_square_vector_system_operator_only(matvec, rhs, tol=tol)

    def solve_column(column):
        return _solve_square_vector_system_operator_only(matvec, column, tol=tol)

    solutions, successes = jax.vmap(
        solve_column,
        in_axes=1,
        out_axes=(1, 0),
    )(rhs)
    return solutions, jnp.all(successes)


def _least_squares_normal_operator(residual_fn, x):
    flat_residual_fn = jax.jit(_flattened_residual_output(residual_fn))
    _, pullback = jax.vjp(flat_residual_fn, x)
    first_leaf = _require_tree_first_leaf(
        x,
        detail="Least-squares linear operator state must contain at least one leaf.",
    )
    dtype = first_leaf.dtype
    decision_size = sum(
        int(np.asarray(jnp.asarray(leaf).size)) for leaf in jax.tree_util.tree_leaves(x)
    )

    def matvec_column(v):
        return _least_squares_matvec(flat_residual_fn, x, pullback, v)

    def matvec(v):
        return _apply_column_batched_operator(matvec_column, v)

    return {
        "kind": "least_squares_normal",
        "shape": (decision_size, decision_size),
        "dtype": dtype,
        "flat_residual_fn": flat_residual_fn,
        "matvec": matvec,
        "transpose_matvec": matvec,
    }


def _solve_least_squares_normal_system(
    residual_fn,
    x,
    rhs,
    *,
    tol,
):
    solution, _ = _solve_least_squares_normal_system_with_status(
        residual_fn,
        x,
        rhs,
        tol=tol,
    )
    return solution


def _solve_least_squares_normal_system_with_status(
    residual_fn,
    x,
    rhs,
    *,
    tol,
):
    operator = _least_squares_normal_operator(residual_fn, x)
    return _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )


def _hessian_linear_operator(objective_fn, x, *, stab=0.0):
    hvp_fn = _hessian_vector_product_fn(objective_fn)
    first_leaf = _require_tree_first_leaf(
        x,
        detail="Hessian linear operator state must contain at least one leaf.",
    )
    dtype = first_leaf.dtype
    decision_size = int(np.asarray(jnp.asarray(x).size))
    stab_value = _optimizer_scalar(stab, dtype=dtype)

    def matvec_column(v):
        return hvp_fn(x, v) + stab_value * v

    def matvec(v):
        return _apply_column_batched_operator(matvec_column, v)

    return {
        "kind": "hessian",
        "shape": (decision_size, decision_size),
        "dtype": dtype,
        "matvec": matvec,
        "transpose_matvec": matvec,
    }


def _solve_hessian_system(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    solution, _ = _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )
    return solution


def _solve_hessian_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    return _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )


def _solve_hessian_least_squares_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    """Solve singular Hessian systems through operator-only normal equations.

    Some LS Boozer fixtures expose gauge-null Hessian directions, so the
    adjoint equation can be inconsistent even with finite branch tangents. The
    target lane uses the Moore-Penrose minimum-residual system for those
    Hessian linearizations while keeping the path matrix-free: solve
    ``H.T @ H @ y = H.T @ rhs`` through the same operator GMRES contract.
    """
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    rhs = jnp.asarray(rhs)
    normal_rhs = operator["transpose_matvec"](rhs)

    def normal_matvec(vector):
        return operator["transpose_matvec"](operator["matvec"](vector))

    solution, normal_success = _solve_square_array_system_operator_only(
        normal_matvec,
        normal_rhs,
        tol=tol,
    )
    normal_residual = normal_rhs - normal_matvec(solution)
    primal_residual = rhs - operator["matvec"](solution)
    success = (
        normal_success
        & jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(normal_residual))
        & jnp.all(jnp.isfinite(primal_residual))
    )
    return solution, success


def _jacobian_linear_operator(residual_fn, x):
    jvp_fn = _jacobian_vector_product_fn(residual_fn)
    residual_x, pullback = jax.vjp(residual_fn, x)
    residual_size = int(np.asarray(jnp.asarray(residual_x).size))
    decision_size = int(np.asarray(jnp.asarray(x).size))
    dtype = jnp.asarray(x).dtype

    def matvec_column(v):
        return jvp_fn(x, v)

    def transpose_matvec_column(v):
        return pullback(v)[0]

    def matvec(v):
        return _apply_column_batched_operator(matvec_column, v)

    def transpose_matvec(v):
        return _apply_column_batched_operator(transpose_matvec_column, v)

    return {
        "kind": "jacobian",
        "shape": (residual_size, decision_size),
        "dtype": dtype,
        "matvec": matvec,
        "transpose_matvec": transpose_matvec,
    }


def _solve_jacobian_system(
    residual_fn,
    x,
    rhs,
    *,
    transpose,
    tol,
):
    operator = _jacobian_linear_operator(residual_fn, x)
    matvec = operator["transpose_matvec"] if transpose else operator["matvec"]
    solution, _ = _solve_square_array_system_operator_only(matvec, rhs, tol=tol)
    return solution


def _solve_jacobian_system_with_status(
    residual_fn,
    x,
    rhs,
    *,
    transpose,
    tol,
):
    operator = _jacobian_linear_operator(residual_fn, x)
    matvec = operator["transpose_matvec"] if transpose else operator["matvec"]
    return _solve_square_array_system_operator_only(matvec, rhs, tol=tol)


def newton_polish(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    materialize_hessian=True,
    max_dense_hessian_bytes=None,
    dense_newton_steps=False,
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

    hessian_size = int(np.asarray(jnp.asarray(x).size))
    dense_step_materialized, dense_step_report = _resolve_dense_hessian_materialization(
        bool(dense_newton_steps),
        hessian_size,
        x.dtype,
        max_dense_hessian_bytes,
    )

    nit = 0
    iterative_refinement_ran = False
    final_step_iterative_refinement_ran = False
    dense_refinement_ran = False
    final_step_dense_refinement_ran = False
    while nit < maxiter and float(norm) > tol:
        linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))
        dense_refine_step = False
        if dense_step_materialized:
            refine_step = float(norm) < 1e-9
            dense_refine_step = refine_step
            H_step = _stabilize_dense_hessian(
                _materialize_dense_hessian(
                    hvp_fn,
                    x,
                    symmetrize=False,
                ),
                stab,
            )
            dx = _solve_dense_newton_step(H_step, grad, refine=refine_step)
            dense_refinement_ran = dense_refinement_ran or refine_step
            iterative_refinement_ran = iterative_refinement_ran or refine_step
        else:
            refine_step = False
            dx, linear_residual, _ = _gmres_solve_newton_system(
                hvp_fn,
                x,
                grad,
                stab=stab,
                tol=linear_tol,
            )
            linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
            if (
                np.all(np.isfinite(np.asarray(dx)))
                and linear_residual_norm > linear_tol
            ):
                correction, _, _ = _gmres_solve_newton_system(
                    hvp_fn,
                    x,
                    linear_residual,
                    stab=stab,
                    tol=linear_tol,
                )
                if np.all(np.isfinite(np.asarray(correction))):
                    dx = dx + correction
                    iterative_refinement_ran = True
                    refine_step = True
        candidate = _backtracking_value_grad_step(
            val_and_grad_fn,
            x,
            dx,
            val,
            grad,
            norm,
        )
        if not bool(candidate["accepted"]):
            break
        x = candidate["x"]
        val = candidate["val"]
        grad = candidate["grad"]
        norm = candidate["norm"]
        nit += 1
        final_step_iterative_refinement_ran = bool(refine_step)
        final_step_dense_refinement_ran = bool(dense_refine_step)
        if progress_callback is not None:
            progress_callback(nit, float(val), float(norm))

    materialize_hessian, dense_report = _resolve_dense_hessian_materialization(
        materialize_hessian,
        hessian_size,
        x.dtype,
        max_dense_hessian_bytes,
    )
    if bool(dense_newton_steps):
        dense_report["dense_newton_steps_materialized"] = dense_step_materialized
        dense_report["dense_newton_steps_message"] = dense_step_report["message"]
    H = None
    if materialize_hessian:
        H = _stabilize_dense_hessian(
            _materialize_dense_hessian(
                hvp_fn,
                x,
                symmetrize=True,
            ),
            stab,
        )

    return {
        "x": x,
        "fun": val,
        "grad": grad,
        "hessian": H,
        "nit": nit,
        "newton_iter": nit,
        "success": bool(float(norm) <= tol),
        "final_gradient_norm": float(norm),
        "final_gradient_inf_norm": float(jnp.linalg.norm(grad, ord=jnp.inf)),
        "iterative_refinement_ran": bool(iterative_refinement_ran),
        "final_step_iterative_refinement_ran": bool(
            final_step_iterative_refinement_ran
        ),
        "dense_refinement_ran": bool(dense_refinement_ran),
        "final_step_dense_refinement_ran": bool(final_step_dense_refinement_ran),
        "hessian_materialized": materialize_hessian,
        **dense_report,
    }


@lru_cache(maxsize=128)
def _make_traceable_newton_polish_runner(
    objective_fn,
    maxiter,
    tol,
    stab,
    materialize_hessian,
    max_dense_hessian_bytes,
    progress_callback,
):
    requested_materialize_hessian = materialize_hessian

    def run_solver(x_init, fn_args):
        def objective_eval(x):
            return objective_fn(x, *fn_args)

        grad_fn = jax.grad(objective_eval)
        val_and_grad_fn = jax.value_and_grad(objective_eval)

        def hvp_fn(x, v):
            return jax.jvp(grad_fn, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = _optimizer_scalar(tol, dtype=dtype)
        val0, grad0 = val_and_grad_fn(x_init)
        norm0 = jnp.linalg.norm(grad0)
        hessian_size = int(np.asarray(jnp.asarray(x_init).size))
        materialize_final_hessian, dense_report = (
            _resolve_dense_hessian_materialization(
                requested_materialize_hessian,
                hessian_size,
                x_init.dtype,
                max_dense_hessian_bytes,
            )
        )

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (state["norm"] > tol_value)
                & (~state["stalled"])
            )

        def body_fun(state):
            stab_value = _optimizer_scalar(stab, dtype=state["x"].dtype)
            linear_tol = _eisenstat_walker_choice2_tolerance(
                state["norm"],
                state["previous_norm"],
                tol=tol_value,
            )

            def matvec(v):
                return hvp_fn(state["x"], v) + stab_value * v

            dx, linear_success = _solve_square_array_system_operator_only(
                matvec,
                state["grad"],
                tol=linear_tol,
            )
            candidate = _backtracking_value_grad_step(
                val_and_grad_fn,
                state["x"],
                dx,
                state["val"],
                state["grad"],
                state["norm"],
            )
            accepted = linear_success & candidate["accepted"]
            next_nit = state["nit"] + 1
            if progress_callback is not None:
                lax.cond(
                    accepted,
                    lambda _: jax.debug.callback(
                        progress_callback,
                        next_nit,
                        candidate["val"],
                        candidate["norm"],
                    ),
                    lambda _: None,
                    operand=None,
                )
            return {
                "x": lax.select(accepted, candidate["x"], state["x"]),
                "val": lax.select(accepted, candidate["val"], state["val"]),
                "grad": lax.select(accepted, candidate["grad"], state["grad"]),
                "norm": lax.select(accepted, candidate["norm"], state["norm"]),
                "previous_norm": lax.select(
                    accepted,
                    state["norm"],
                    state["previous_norm"],
                ),
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
                "previous_norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
                "stalled": jnp.asarray(False),
            },
        )

        val_final, grad_final = val_and_grad_fn(state["x"])
        norm_final = jnp.linalg.norm(grad_final)
        H = None
        if materialize_final_hessian:
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
            "hessian_materialized": materialize_final_hessian,
            **dense_report,
        }

    run_solver.__name__ = "traceable_newton_polish_run_solver"
    return jax.jit(run_solver)


def newton_polish_traceable(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
    materialize_hessian=True,
    max_dense_hessian_bytes=None,
    progress_callback=None,
    args=(),
):
    """Trace-safe Newton polish for JAX-traceable objective paths.

    This variant keeps all loop state and step decisions inside JAX control
    flow so higher-level traced objectives can invoke the Newton stage without
    crossing back into Python. Newton corrections use the operator-only GMRES
    path; the dense Hessian policy only controls final compatibility metadata.
    """
    runner = _make_traceable_newton_polish_runner(
        objective_fn,
        int(maxiter),
        float(tol),
        float(stab),
        bool(materialize_hessian),
        max_dense_hessian_bytes,
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
    hot loop. The dense Jacobian is rebuilt once at the final iterate only for
    public compatibility metadata and diagnostics.
    """
    res_fn = jax.jit(residual_fn)
    jvp_fn = _jacobian_vector_product_fn(residual_fn)

    x = x0
    r = res_fn(x)
    norm = jnp.linalg.norm(r)
    linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))

    nit = 0
    exact_newton_linear_residual_rel = None
    exact_refinement_correction_rel = None
    while nit < maxiter and float(norm) > tol:
        dx, linear_residual, _ = _gmres_solve_exact_newton_system(
            jvp_fn,
            x,
            r,
            tol=linear_tol,
        )
        dx_before_refinement = dx
        exact_newton_linear_residual_rel = float(
            _relative_residual_norm(linear_residual, r)
        )
        linear_residual_norm = float(np.linalg.norm(np.asarray(linear_residual)))
        if not np.all(np.isfinite(np.asarray(dx))):
            break
        if linear_residual_norm > linear_tol:
            correction, _, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                x,
                linear_residual,
                tol=linear_tol,
            )
            if np.all(np.isfinite(np.asarray(correction))):
                dx = dx + correction
                denominator = np.linalg.norm(np.asarray(dx_before_refinement))
                exact_refinement_correction_rel = float(
                    np.linalg.norm(np.asarray(correction)) / max(denominator, 1e-30)
                )
        x_candidate = x - dx
        r_candidate = res_fn(x_candidate)
        norm_candidate = jnp.linalg.norm(r_candidate)
        if float(norm_candidate) <= float(norm):
            x = x_candidate
            r = r_candidate
            norm = norm_candidate
        else:
            break
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
            "exact_newton_linear_residual_rel": exact_newton_linear_residual_rel,
            "exact_refinement_correction_rel": exact_refinement_correction_rel,
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
        "exact_newton_linear_residual_rel": exact_newton_linear_residual_rel,
        "exact_refinement_correction_rel": exact_refinement_correction_rel,
        **report,
    }


@lru_cache(maxsize=128)
def _make_traceable_exact_newton_runner(
    residual_fn,
    maxiter,
    tol,
):
    def run_solver(x_init, fn_args):
        def residual_eval(x):
            return residual_fn(x, *fn_args)

        def jvp_fn(x, v):
            return jax.jvp(residual_eval, (x,), (v,))[1]

        dtype = jnp.asarray(x_init).dtype
        tol_value = _optimizer_scalar(tol, dtype=dtype)
        r0 = residual_eval(x_init)
        norm0 = jnp.linalg.norm(r0)

        def cond_fun(state):
            return (
                (state["nit"] < maxiter)
                & (state["norm"] > tol_value)
                & (~state["stalled"])
            )

        def body_fun(state):
            linear_tol_iteration = _eisenstat_walker_choice2_tolerance(
                state["norm"],
                state["previous_norm"],
                tol=tol_value,
            )
            dx, linear_residual, _ = _gmres_solve_exact_newton_system(
                jvp_fn,
                state["x"],
                state["residual"],
                tol=linear_tol_iteration,
            )
            linear_residual_norm = jnp.linalg.norm(linear_residual)
            linear_residual_rel = _relative_residual_norm(
                linear_residual,
                state["residual"],
            )

            def add_correction(current_dx):
                correction, _, _ = _gmres_solve_exact_newton_system(
                    jvp_fn,
                    state["x"],
                    linear_residual,
                    tol=linear_tol_iteration,
                )
                correction_rel = jnp.linalg.norm(correction) / jnp.maximum(
                    jnp.linalg.norm(current_dx),
                    _device_scalar(
                        jnp.finfo(current_dx.dtype).tiny,
                        dtype=current_dx.dtype,
                    ),
                )
                correction_finite = jnp.all(jnp.isfinite(correction))
                return (
                    lax.cond(
                        correction_finite,
                        lambda corr: current_dx + corr,
                        lambda _corr: current_dx,
                        correction,
                    ),
                    lax.select(
                        correction_finite,
                        correction_rel,
                        _device_scalar(jnp.nan, dtype=current_dx.dtype),
                    ),
                )

            dx, correction_rel = lax.cond(
                jnp.all(jnp.isfinite(dx))
                & (linear_residual_norm > linear_tol_iteration),
                add_correction,
                lambda current_dx: (
                    current_dx,
                    _device_scalar(0.0, dtype=current_dx.dtype),
                ),
                dx,
            )
            candidate = _backtracking_residual_step(
                residual_eval,
                state["x"],
                dx,
                state["residual"],
                state["norm"],
            )
            accepted = candidate["accepted"]
            return {
                "x": lax.select(accepted, candidate["x"], state["x"]),
                "residual": lax.select(
                    accepted,
                    candidate["residual"],
                    state["residual"],
                ),
                "norm": lax.select(accepted, candidate["norm"], state["norm"]),
                "previous_norm": lax.select(
                    accepted,
                    state["norm"],
                    state["previous_norm"],
                ),
                "nit": lax.select(accepted, state["nit"] + 1, state["nit"]),
                "stalled": ~accepted,
                "exact_newton_linear_residual_rel": lax.select(
                    accepted,
                    linear_residual_rel,
                    state["exact_newton_linear_residual_rel"],
                ),
                "exact_refinement_correction_rel": lax.select(
                    accepted,
                    correction_rel,
                    state["exact_refinement_correction_rel"],
                ),
            }

        state = lax.while_loop(
            cond_fun,
            body_fun,
            {
                "x": x_init,
                "residual": r0,
                "norm": norm0,
                "previous_norm": norm0,
                "nit": jnp.asarray(0, dtype=jnp.int32),
                "stalled": jnp.asarray(False),
                "exact_newton_linear_residual_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
                "exact_refinement_correction_rel": jnp.asarray(
                    jnp.nan,
                    dtype=dtype,
                ),
            },
        )
        return {
            "x": state["x"],
            "residual": state["residual"],
            "nit": state["nit"],
            "success": state["norm"] <= tol_value,
            "exact_newton_linear_residual_rel": state[
                "exact_newton_linear_residual_rel"
            ],
            "exact_refinement_correction_rel": state["exact_refinement_correction_rel"],
        }

    run_solver.__name__ = "traceable_exact_newton_run_solver"
    return jax.jit(run_solver)


def newton_exact_traceable(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
    args=(),
):
    """Trace-safe Newton solver for the exact Boozer residual system.

    The loop keeps Jacobian application matrix-free via JVPs and does not
    materialize dense Jacobians. Public dense metadata belongs to
    ``newton_exact(...)`` / ``BoozerSurfaceJAX.run_code()``.
    """
    normalized_args = _normalize_solver_args(args)
    runner = _make_traceable_exact_newton_runner(
        residual_fn,
        int(maxiter),
        float(tol),
    )
    result = runner(x0, normalized_args)
    result["jacobian"] = None
    result["jacobian_materialized"] = False
    result["failure_category"] = None
    result["failure_stage"] = None
    result["message"] = None
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


def _least_squares_state_to_optimize_result(result):
    nit = int(_host_scalar(result["nit"], dtype=np.int64))
    status = int(_host_scalar(result["status"], dtype=np.int64))
    info = int(_host_scalar(result["info"], dtype=np.int64))
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
        info=info,
        success=success,
        message=_least_squares_result_message(
            status,
            success,
            info=info,
        ),
        dense_linearization_materialized=result["dense_linearization_materialized"],
        dense_linearization_kind=result.get("dense_linearization_kind"),
        dense_residual_jacobian_shape=result.get("dense_residual_jacobian_shape"),
        dense_residual_jacobian_bytes=result.get("dense_residual_jacobian_bytes"),
        dense_hessian_shape=result.get("dense_hessian_shape"),
        dense_hessian_bytes=result.get("dense_hessian_bytes"),
        dense_linearization_bytes=result.get("dense_linearization_bytes"),
        max_dense_linearization_bytes=result.get("max_dense_linearization_bytes"),
        failure_category=result.get("failure_category"),
        failure_stage=result.get("failure_stage"),
        optimistix_result=result.get("optimistix_result"),
        optimistix_result_message=result.get("optimistix_result_message"),
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
    if method not in _TARGET_LEAST_SQUARES_METHODS:
        raise ValueError(
            "target_least_squares() only supports method='lm-ondevice', "
            "method='lm-minpack-ondevice', or method='optimistix-lm-ondevice'. "
            f"Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")
    ftol = options.get("ftol", _DEFAULT_LM_FTOL)
    xtol = options.get("xtol", _DEFAULT_LM_XTOL)
    materialize_dense_linearization = bool(
        options.get("materialize_dense_linearization", True)
    )
    max_dense_linearization_bytes = options.get("max_dense_linearization_bytes")
    callback = options.get("callback")
    progress_callback = options.get("progress_callback")
    if method == "optimistix-lm-ondevice":
        result = jax_least_squares_optimistix(
            residual_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            ftol=ftol,
            xtol=xtol,
            gtol=options.get("gtol"),
            materialize_dense_linearization=materialize_dense_linearization,
            max_dense_linearization_bytes=max_dense_linearization_bytes,
            callback=callback,
            progress_callback=progress_callback,
        )
        return _least_squares_state_to_optimize_result(result)

    solver = (
        levenberg_marquardt_minpack_traceable
        if method == "lm-minpack-ondevice"
        else levenberg_marquardt_traceable
    )
    gtol = options.get("gtol")
    if method == "lm-minpack-ondevice" and gtol is None:
        gtol = 1e-8
    result = solver(
        residual_fn,
        x0,
        maxiter=maxiter,
        tol=tol,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        materialize_dense_linearization=materialize_dense_linearization,
        max_dense_linearization_bytes=max_dense_linearization_bytes,
        callback=callback,
        progress_callback=progress_callback,
    )

    return _least_squares_state_to_optimize_result(result)


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
    failure_callback=None,
    initial_value_and_grad=None,
):
    """Explicit CPU/reference scalar optimizer entrypoint."""
    if failure_callback is not None and method not in _REFERENCE_TRACE_METHODS:
        raise ValueError(
            "reference_minimize() only supports failure_callback for "
            "method='lbfgs-trace'."
        )
    if initial_value_and_grad is not None and (
        method not in _REFERENCE_TRACE_METHODS or not value_and_grad
    ):
        raise ValueError(
            "reference_minimize() only supports initial_value_and_grad for "
            "explicit value-and-gradient objectives with method='lbfgs-trace'."
        )
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
        failure_callback=failure_callback,
        initial_value_and_grad=initial_value_and_grad,
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
    initial_value_and_grad=None,
):
    """Explicit JAX target scalar optimizer entrypoint."""
    options = dict(options or {})
    if failure_callback is not None:
        raise ValueError(
            "target_minimize() does not support failure_callback. "
            "Use reference_minimize(method='lbfgs-trace') for host-side "
            "L-BFGS rejection diagnostics."
        )
    if initial_value_and_grad is not None and (
        method != "lbfgs-ondevice" or not value_and_grad
    ):
        raise ValueError(
            "target_minimize() only supports initial_value_and_grad for "
            "explicit value-and-gradient objectives with method='lbfgs-ondevice'."
        )
    if method in _TARGET_LBFGSB_METHODS:
        unsupported_options = _UNSUPPORTED_TARGET_LBFGSB_OPTIONS.intersection(options)
        if unsupported_options:
            raise ValueError(
                "target L-BFGS-B methods follow SciPy L-BFGS-B options and do "
                f"not support {sorted(unsupported_options)}."
            )
    if method in _TARGET_SCIPY_CONTROL_METHODS:
        if not value_and_grad:
            raise RuntimeError(
                f"target_minimize() requires value_and_grad=True for method={method!r}."
            )
        fun = wrap_strict_target_lane_value_and_grad(fun)
        fun, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
            fun,
            x0,
            value_and_grad=True,
            callback=callback,
        )
        if callback is not None:
            options["callback"] = callback
        if progress_callback is not None:
            options["progress_callback"] = progress_callback
        required_backend = (
            "scipy-jax-fullgraph"
            if method == "lbfgs-scipy-jax-fullgraph"
            else "scipy-jax"
        )
        require_target_backend_x64(required_backend)
        reference_optimizer = _load_reference_optimizer_module()
        result = reference_optimizer.target_scipy_minimize_value_and_grad(
            fun,
            x0,
            method="lbfgs",
            tol=tol,
            maxiter=maxiter,
            options=options,
        )
        return _finalize_optimizer_result(result, pytree_adapter)
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

    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    require_target_backend_x64("ondevice")

    # All remaining methods require the private optimizer package.
    _require_private_package(method)
    if method == "lbfgs-ondevice":
        lbfgs_ftol = float(options.get("ftol", tol))

    if value_and_grad:
        if method != "lbfgs-ondevice":
            raise RuntimeError(
                "Explicit value-and-gradient objectives are only supported on the "
                "trusted SciPy reference methods and lbfgs-ondevice today."
            )
        fun = wrap_strict_target_lane_value_and_grad(fun)
        state = _minimize_lbfgs_private_value_and_grad(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 10)),
            ftol=lbfgs_ftol,
            maxfun=options.get("maxfun"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
            initial_value_and_grad=initial_value_and_grad,
            record_optimizer_state_trace=bool(
                options.get("record_optimizer_state_trace", False)
            ),
            max_optimizer_state_trace_bytes=options.get(
                "max_optimizer_state_trace_bytes"
            ),
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
            maxcor=int(options.get("maxcor", 10)),
            ftol=lbfgs_ftol,
            maxfun=options.get("maxfun"),
            maxls=int(options.get("maxls", 20)),
            callback=options.get("callback"),
            progress_callback=options.get("progress_callback"),
            record_optimizer_state_trace=bool(
                options.get("record_optimizer_state_trace", False)
            ),
            max_optimizer_state_trace_bytes=options.get(
                "max_optimizer_state_trace_bytes"
            ),
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

    if method in _REFERENCE_METHODS | _REFERENCE_TRACE_METHODS | _REFERENCE_JAX_METHODS:
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
