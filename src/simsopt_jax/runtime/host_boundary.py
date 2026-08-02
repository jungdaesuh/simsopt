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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, TypeVar

import jax
import numpy as np
from numpy.typing import NDArray

from simsopt_jax.backend.dtypes import runtime_device_put
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    CertificateProbeKeyData,
)

_TreeT = TypeVar("_TreeT")


@dataclass(frozen=True)
class HostTransferSummary:
    """Aggregated device-to-host materializations for one execution phase."""

    phase: str
    calls: int
    leaves: int
    bytes: int


@dataclass
class HostTransferAudit:
    """Context-local ledger for explicit host-boundary measurements.

    The ledger is opt-in and records only JAX array leaves passed through
    :func:`host_value`; ordinary NumPy values do not count as device transfers.
    ``ContextVar`` ownership keeps concurrent solver calls independent.
    """

    _events: list[tuple[str, int, int]] = field(default_factory=list)
    _phases: list[str] = field(default_factory=list)

    def register_phase(self, phase: str) -> None:
        if phase not in self._phases:
            self._phases.append(phase)

    def record(self, phase: str, *, leaves: int, bytes: int) -> None:
        if leaves > 0:
            self._events.append((phase, leaves, bytes))

    def summary(self) -> tuple[HostTransferSummary, ...]:
        aggregates: dict[str, list[int]] = {phase: [0, 0, 0] for phase in self._phases}
        for phase, leaves, bytes in self._events:
            aggregate = aggregates.setdefault(phase, [0, 0, 0])
            aggregate[0] += 1
            aggregate[1] += leaves
            aggregate[2] += bytes
        return tuple(
            HostTransferSummary(
                phase=phase,
                calls=values[0],
                leaves=values[1],
                bytes=values[2],
            )
            for phase, values in aggregates.items()
        )


_ACTIVE_TRANSFER_AUDIT: ContextVar[HostTransferAudit | None] = ContextVar(
    "simsopt_jax_active_host_transfer_audit",
    default=None,
)
_ACTIVE_TRANSFER_PHASE: ContextVar[str] = ContextVar(
    "simsopt_jax_active_host_transfer_phase",
    default="unclassified",
)


@contextmanager
def host_transfer_audit() -> Iterator[HostTransferAudit]:
    """Collect explicit JAX device-to-host transfers in the current context."""

    audit = HostTransferAudit()
    token = _ACTIVE_TRANSFER_AUDIT.set(audit)
    try:
        yield audit
    finally:
        _ACTIVE_TRANSFER_AUDIT.reset(token)


@contextmanager
def host_transfer_phase(phase: str) -> Iterator[None]:
    """Attribute host materializations to a named solver phase."""

    normalized_phase = str(phase)
    audit = _ACTIVE_TRANSFER_AUDIT.get()
    if audit is None:
        yield
        return
    audit.register_phase(normalized_phase)
    token = _ACTIVE_TRANSFER_PHASE.set(normalized_phase)
    try:
        yield
    finally:
        _ACTIVE_TRANSFER_PHASE.reset(token)


def _record_host_transfer(value: object) -> None:
    audit = _ACTIVE_TRANSFER_AUDIT.get()
    if audit is None:
        return
    leaves = 0
    total_bytes = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if not isinstance(leaf, jax.Array):
            continue
        leaves += 1
        total_bytes += (
            int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize
        )
    audit.record(
        _ACTIVE_TRANSFER_PHASE.get(),
        leaves=leaves,
        bytes=total_bytes,
    )


def host_value(value: _TreeT) -> _TreeT:
    """Materialize a JAX value or pytree while preserving its Python structure."""
    _record_host_transfer(value)
    return jax.device_get(value)


def host_array(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> np.ndarray:
    """Materialize ``value`` to a writeable NumPy array at an explicit D2H boundary."""
    array = np.asarray(host_value(value))
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return array


def host_scalar(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> object:
    return host_array(value, dtype=dtype).item()


def host_float(
    value: object,
    *,
    dtype: jax.typing.DTypeLike = np.float64,
) -> float:
    return float(host_scalar(value, dtype=dtype))


def block_until_ready(value: _TreeT) -> _TreeT:
    """Wait for every JAX leaf and return the same pytree structure and values."""
    return jax.block_until_ready(value)


def host_array_after_ready(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> np.ndarray:
    """Wait for device completion, then materialize via :func:`host_array`.

    Distinct from plain :func:`host_array`: solver packaging and callback
    paths need a completion barrier before D2H so timings and host-side
    consumers see finished values.
    """
    return host_array(block_until_ready(value), dtype=dtype)


def host_float_after_ready(
    value: object,
    *,
    dtype: jax.typing.DTypeLike = np.float64,
) -> float:
    """Wait for device completion, then materialize a Python float."""
    return host_float(block_until_ready(value), dtype=dtype)


def host_float64(value, *, has_jax: bool = True) -> np.ndarray:
    """Materialize a value as a NumPy float64 array at an explicit host boundary."""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if not has_jax:
        return np.asarray(value, dtype=np.float64)
    return host_array(value, dtype=np.float64)


def host_int(
    value: object,
    *,
    dtype: jax.typing.DTypeLike = np.int64,
) -> int:
    return int(host_scalar(value, dtype=dtype))


def host_bool(value: object) -> bool:
    return bool(host_scalar(value, dtype=np.bool_))


def host_all_finite(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> bool:
    return bool(np.all(np.isfinite(host_array(value, dtype=dtype))))


def host_inf_norm(
    value: object,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> float:
    return float(np.max(np.abs(host_array(value, dtype=dtype))))


def host_tree(value, *, dtype=None):
    materialized = host_value(value)

    def _hostify_leaf(leaf):
        if isinstance(leaf, jax.core.Tracer):
            return leaf
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.array(leaf, dtype=leaf_dtype, copy=True)
        if dtype is not None and (isinstance(leaf, np.generic) or np.isscalar(leaf)):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree.map(_hostify_leaf, materialized)


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
