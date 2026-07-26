"""Device-to-host materialization SSOT for JAX-backed compatibility lanes.

Ownership split (do not reimplement these patterns in adapters):

* ``simsopt_jax.runtime.host_boundary`` — **host materialization** (D2H):
  ``host_array``, ``host_scalar``, ``host_float``, ``host_tree``, and the
  ready variants that ``block_until_ready`` before materializing.
* ``simsopt_jax.backend.dtypes`` — **device placement** (H2D / on-device cast):
  policy ``runtime_device_put`` / ``as_runtime_array`` / ``as_compute_array``,
  and exact-dtype ``explicit_device_array`` (preserves requested float dtype).

Local thin wrappers that only re-export the same semantics should import from
these owners rather than re-coding ``device_get`` / ``device_put``.
"""

from __future__ import annotations

from typing import TypeVar

import jax
import numpy as np
from numpy.typing import NDArray

from simsopt_jax.backend.dtypes import runtime_device_put
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    CertificateProbeKeyData,
)

_TreeT = TypeVar("_TreeT")


def host_value(value: _TreeT) -> _TreeT:
    """Materialize a JAX value or pytree while preserving its Python structure."""
    return jax.device_get(value)


def host_array(value, *, dtype=None):
    """Materialize ``value`` to a writeable NumPy array at an explicit D2H boundary."""
    array = np.asarray(host_value(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def host_scalar(value, *, dtype=None):
    return host_array(value, dtype=dtype).item()


def host_float(value, *, dtype=np.float64) -> float:
    return float(host_scalar(value, dtype=dtype))


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values."""
    return jax.block_until_ready(value)


def host_array_after_ready(value, *, dtype=None):
    """Wait for device completion, then materialize via :func:`host_array`.

    Distinct from plain :func:`host_array`: solver packaging and callback
    paths need a completion barrier before D2H so timings and host-side
    consumers see finished values.
    """
    return host_array(block_until_ready(value), dtype=dtype)


def host_float_after_ready(value, *, dtype=np.float64) -> float:
    """Wait for device completion, then materialize a Python float."""
    return host_float(block_until_ready(value), dtype=dtype)


def host_float64(value, *, has_jax: bool = True) -> np.ndarray:
    """Materialize a value as a NumPy float64 array at an explicit host boundary."""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if not has_jax:
        return np.asarray(value, dtype=np.float64)
    return host_array(value, dtype=np.float64)


def host_int(value, *, dtype=np.int64) -> int:
    return int(host_scalar(value, dtype=dtype))


def host_bool(value) -> bool:
    return bool(host_scalar(value, dtype=np.bool_))


def host_all_finite(value, *, dtype=None) -> bool:
    return bool(np.all(np.isfinite(host_array(value, dtype=dtype))))


def host_inf_norm(value, *, dtype=None) -> float:
    return float(np.max(np.abs(host_array(value, dtype=dtype))))


def host_tree(value, *, dtype=None):
    def _hostify_leaf(leaf):
        if isinstance(leaf, jax.core.Tracer):
            return leaf
        if isinstance(leaf, jax.Array):
            leaf_dtype = np.dtype(leaf.dtype) if dtype is None else dtype
            return host_array(leaf, dtype=leaf_dtype)
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.asarray(leaf, dtype=leaf_dtype)
        if dtype is not None and (isinstance(leaf, np.generic) or np.isscalar(leaf)):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree.map(_hostify_leaf, value)


def host_tree_after_ready(value, *, dtype=None):
    """Wait for a pytree, then materialize its array leaves on the host."""
    return host_tree(block_until_ready(value), dtype=dtype)


def scalar_pullback_seed(value):
    # Build the pullback seed from ``value`` itself so the scalar cotangent stays
    # on-device under ``jax.transfer_guard("disallow")``.
    always_true = jax.numpy.logical_or(
        jax.numpy.equal(value, value),
        jax.numpy.not_equal(value, value),
    )
    return always_true.astype(value.dtype)


def explicit_cotangent_basis(length: int, index: int, *, dtype):
    basis = np.zeros(int(length), dtype=np.dtype(dtype))
    basis[int(index)] = 1.0
    return runtime_device_put(basis, dtype=dtype)


def runtime_certificate_probe_key(key_data: CertificateProbeKeyData) -> jax.Array:
    """Construct a typed Threefry certificate key on the runtime device."""
    key_words: NDArray[np.uint32] = np.asarray(key_data.words, dtype=np.uint32)
    device_key_data = runtime_device_put(key_words, dtype=np.uint32)
    return jax.random.wrap_key_data(
        device_key_data,
        impl=MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    )
