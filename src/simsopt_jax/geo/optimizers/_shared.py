"""Shared optimizer helpers used by public and private optimizer modules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from simsopt_jax.backend.dtypes import explicit_device_array, runtime_device_put
from simsopt_jax.geo.optimizers._evaluation_provider import (
    TargetScipyDeviceEvaluation,
    TargetScipyDevicePacketLayout,
    TargetScipyEvaluationProvider,
    TargetScipyHostEvaluation,
    TargetScipyHostFinalizer,
    _require_host_numeric_leaf,
)
from simsopt_jax.runtime.host_boundary import host_array

PRIVATE_OPTIMIZER_JAX_VERSION = "0.10.0"
_CACHEABLE_VALUE_AND_GRAD_ATTR = "_simsopt_cache_jit_value_and_grad"
_STRUCTURED_SOLVER_CACHE_TOKEN_ATTR = "_simsopt_structured_solver_cache_token"
_CACHEABLE_LINEAR_OPERATOR_ATTR = "_simsopt_cache_jit_linear_operator"


_DecisionTreeT = TypeVar("_DecisionTreeT")
_DevicePacketT = TypeVar("_DevicePacketT")
_HostPacketT = TypeVar("_HostPacketT")
_DeviceValueT = TypeVar("_DeviceValueT")
_DeviceGradientTreeT = TypeVar("_DeviceGradientTreeT")
_HostValueT = TypeVar("_HostValueT")
_HostGradientTreeT = TypeVar("_HostGradientTreeT")


def _version_key(raw_version: str) -> tuple[int, ...]:
    base = raw_version.split("+", 1)[0]
    parts = re.findall(r"\d+", base)
    return tuple(int(part) for part in parts)


def private_optimizer_runtime_is_supported(version: str) -> bool:
    """Return True when the runtime is newer than the pinned optimizer source."""
    return _version_key(version) >= _version_key(PRIVATE_OPTIMIZER_JAX_VERSION)


def _x64_enabled():
    return bool(jax.config.jax_enable_x64)


def cast_floating_tree(tree, dtype):
    """Cast floating array leaves while preserving discrete metadata."""
    resolved_dtype = np.dtype(dtype)

    def cast_leaf(leaf):
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is not None and np.issubdtype(np.dtype(leaf_dtype), np.floating):
            return jnp.asarray(leaf, dtype=resolved_dtype)
        return leaf

    return jax.tree.map(cast_leaf, tree)


def mark_cacheable_jit_value_and_grad(fun):
    """Mark a mutable callable for shared value/gradient and operator JIT caches."""
    setattr(fun, _CACHEABLE_VALUE_AND_GRAD_ATTR, True)
    setattr(fun, _CACHEABLE_LINEAR_OPERATOR_ATTR, True)
    return fun


def _optimizer_flat_vector(value, *, dtype=None) -> jax.Array:
    if isinstance(value, jax.Array):
        if dtype is None or value.dtype == dtype:
            return value
        return jnp.asarray(value, dtype=dtype)
    if hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return explicit_device_array(value, dtype=dtype)


def _optimizer_scalar(value, *, dtype) -> jax.Array:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    return runtime_device_put(value, dtype=dtype)


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
        array = host_array(leaf)
    elif isinstance(leaf, (np.ndarray, np.generic)) or np.isscalar(leaf):
        array = np.asarray(leaf)
    else:
        return leaf
    if array.ndim == 0:
        return array.item()
    return array


def _hostify_optimizer_tree(value):
    return jax.tree.map(_hostify_optimizer_leaf, value)


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


@dataclass(frozen=True)
class _OptimizerPytreeAdapter(Generic[_DecisionTreeT]):
    flat_dtype: np.dtype
    unravel: Callable[[jax.Array], _DecisionTreeT]
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

        if isinstance(fun, TargetScipyEvaluationProvider):
            return _PytreeTargetScipyEvaluationProvider(
                adapter=self,
                provider=fun,
            )

        def wrapped_value_and_grad(flat_x):
            flat_x = _optimizer_flat_vector(flat_x)
            value, grad_tree = fun(self.unravel(flat_x))
            return self.flatten_device_value_and_gradient(
                flat_x,
                value,
                grad_tree,
            )

        return wrapped_value_and_grad

    def flatten_device_value_and_gradient(self, flat_x, value, grad_tree):
        flat_x = _optimizer_flat_vector(flat_x)
        _, grad_tree_def = jax.tree.flatten(grad_tree)
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

    def flatten_host_gradient(self, gradient_tree) -> np.ndarray:
        leaves, tree_def = jax.tree.flatten(gradient_tree)
        if tree_def != self.tree_def:
            raise ValueError(
                "Explicit value-and-gradient objectives must return a gradient "
                "with the same pytree structure as x0."
            )
        for leaf in leaves:
            _require_host_numeric_leaf(leaf)
        host_leaves = tuple(np.asarray(leaf, dtype=self.flat_dtype) for leaf in leaves)
        for leaf, (expected_shape, _expected_dtype) in zip(
            host_leaves,
            self.leaf_signature,
        ):
            if leaf.shape != expected_shape:
                raise ValueError(
                    "Explicit value-and-gradient objectives must return gradient "
                    "leaves matching the corresponding x0 shapes."
                )
        if not host_leaves:
            return np.empty((0,), dtype=self.flat_dtype)
        return np.concatenate(tuple(leaf.reshape(-1) for leaf in host_leaves))

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


@dataclass(frozen=True)
class _PytreeTargetScipyHostFinalizer(
    Generic[_DecisionTreeT, _HostPacketT, _HostValueT, _HostGradientTreeT]
):
    adapter: _OptimizerPytreeAdapter[_DecisionTreeT]
    finalize_host: TargetScipyHostFinalizer[
        _HostPacketT,
        _HostValueT,
        _HostGradientTreeT,
    ]

    def __call__(
        self,
        host_decision_vector: np.ndarray,
        host_packet: _HostPacketT,
        device_layout: TargetScipyDevicePacketLayout,
        /,
    ) -> TargetScipyHostEvaluation[_HostValueT, np.ndarray]:
        evaluation = self.finalize_host(
            host_decision_vector,
            host_packet,
            device_layout,
        )
        return TargetScipyHostEvaluation(
            value=evaluation.value,
            gradient=self.adapter.flatten_host_gradient(evaluation.gradient),
            lifecycle=evaluation.lifecycle,
        )


@dataclass(frozen=True)
class _PytreeTargetScipyEvaluationProvider(
    Generic[
        _DecisionTreeT,
        _DevicePacketT,
        _HostPacketT,
        _DeviceValueT,
        _DeviceGradientTreeT,
        _HostValueT,
        _HostGradientTreeT,
    ]
):
    adapter: _OptimizerPytreeAdapter[_DecisionTreeT]
    provider: TargetScipyEvaluationProvider[
        _DecisionTreeT,
        _DevicePacketT,
        _HostPacketT,
        _DeviceValueT,
        _DeviceGradientTreeT,
        _HostValueT,
        _HostGradientTreeT,
    ]

    def __call__(
        self,
        flat_decision_vector: jax.Array,
        /,
    ) -> tuple[jax.Array, jax.Array]:
        flat_decision_vector = _optimizer_flat_vector(flat_decision_vector)
        value, gradient = self.provider(self.adapter.unravel(flat_decision_vector))
        return self.adapter.flatten_device_value_and_gradient(
            flat_decision_vector,
            value,
            gradient,
        )

    def evaluate_target_scipy(
        self,
        flat_decision_vector: jax.Array,
        host_decision_vector: np.ndarray,
        /,
    ) -> TargetScipyDeviceEvaluation[
        _DevicePacketT,
        _HostPacketT,
        _HostValueT,
        np.ndarray,
    ]:
        flat_decision_vector = _optimizer_flat_vector(flat_decision_vector)
        evaluation = self.provider.evaluate_target_scipy(
            self.adapter.unravel(flat_decision_vector),
            host_decision_vector,
        )
        return TargetScipyDeviceEvaluation(
            device_packet=evaluation.device_packet,
            finalize_host=_PytreeTargetScipyHostFinalizer(
                adapter=self.adapter,
                finalize_host=evaluation.finalize_host,
            ),
        )


def _prepare_optimizer_pytree_adapter(x0):
    if _is_flat_optimizer_vector(x0):
        return None
    leaves, tree_def = jax.tree.flatten(x0)
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


_OptimizerCallable = Callable[..., object]
_PreparedOptimizerInputs = tuple[
    _OptimizerCallable,
    object,
    object | None,
    _OptimizerPytreeAdapter[object] | None,
]


def _prepare_optimizer_callable_inputs(
    fun: _OptimizerCallable,
    x0: object,
    *,
    value_and_grad: bool,
    callback: object | None,
) -> _PreparedOptimizerInputs:
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
