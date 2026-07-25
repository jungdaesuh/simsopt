"""Shared runtime/host boundary helpers for JAX-backed compatibility lanes."""

from __future__ import annotations

import jax
import numpy as np
from numpy.typing import NDArray

from simsopt_jax.backend.dtypes import runtime_device_put
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    CertificateProbeKeyData,
)


def host_array(value, *, dtype=None):
    with jax.transfer_guard_device_to_host("allow"):
        array = np.asarray(jax.device_get(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def host_scalar(value, *, dtype=None):
    return host_array(value, dtype=dtype).item()


def host_float(value, *, dtype=np.float64) -> float:
    return float(host_scalar(value, dtype=dtype))


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
