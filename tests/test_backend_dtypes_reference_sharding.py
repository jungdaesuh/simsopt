"""Regression: ``_reference_sharding`` must short-circuit on JAX tracers.

A tracer is a ``jax.Array`` but carries no concrete sharding. The prior code
probed ``tracer.sharding`` via ``getattr(..., None)``; that attribute access
raises ``AttributeError`` whose message eagerly walks the entire jaxpr (jax's
``_origin_msg``/``find_progenitors``) only to be discarded. Paid once per
``as_runtime_array`` call across an O(jaxpr) trace, that is an O(jaxpr^2)
construction cost that scaled with resolution (the single-stage
``value_and_grad`` build wedge). The guard returns ``None`` for tracers without
probing; ``as_runtime_array`` already bypasses reference placement for traced
values, so behavior is unchanged.
"""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend


def test_reference_sharding_short_circuits_on_tracer():
    """On a tracer the sharding-compat path is never reached (the O(jaxpr) walk)."""
    captured: dict[str, object] = {}

    @jax.jit
    def f(x):
        with mock.patch.object(dtypes, "_compatible_reference_sharding") as compat:
            captured["result"] = dtypes._reference_sharding(x, ndim=1)
            captured["compat_calls"] = compat.call_count
        return x

    f(jnp.zeros(3))

    # Old behavior: probed tracer.sharding (-> None after the jaxpr walk) then
    # called _compatible_reference_sharding(None, ...). The guard returns None
    # first, so the compat path is never invoked for a tracer.
    assert captured["result"] is None
    assert captured["compat_calls"] == 0


def test_reference_sharding_still_probes_concrete_array():
    """A concrete (non-traced) array is unaffected: it is still probed."""
    arr = jnp.zeros(3)
    with mock.patch.object(
        dtypes,
        "_compatible_reference_sharding",
        wraps=dtypes._compatible_reference_sharding,
    ) as compat:
        result = dtypes._reference_sharding(arr, ndim=1)

    # The concrete array goes through the probe; a single-device sharding is not
    # a NamedSharding, so the compatible result is None -- but the path runs.
    assert compat.call_count == 1
    assert result is None


def test_reference_sharding_handles_tracer_leaf_in_sequence():
    """The list/tuple branch returns None for a tracer leaf and does not crash.

    This is a correctness smoke for the leaf-skip edit, not the regression guard
    (the old list branch also fell through to None for a tracer leaf via
    ``getattr(leaf, "sharding", None)`` -> None); the O(jaxpr) cost the fix
    removes is pinned for the scalar case by the first test.
    """
    captured: dict[str, object] = {}

    @jax.jit
    def f(x):
        with mock.patch.object(dtypes, "_compatible_reference_sharding") as compat:
            captured["result"] = dtypes._reference_sharding([x], ndim=1)
            captured["compat_calls"] = compat.call_count
        return x

    f(jnp.zeros(3))

    assert captured["result"] is None
    assert captured["compat_calls"] == 0


def test_runtime_device_put_uses_runtime_device_when_no_target(monkeypatch):
    """Implicit placement follows the runtime policy device, not JAX defaults."""
    runtime_device = object()
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert placement is runtime_device
    assert placements == [runtime_device]


def test_runtime_device_put_preserves_explicit_target(monkeypatch):
    """Explicit target/sharding placement still takes precedence."""
    explicit_target = object()
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    def _unexpected_runtime_device():
        raise AssertionError("explicit placement must not query runtime device")

    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", _unexpected_runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3], target=explicit_target)

    assert isinstance(array, np.ndarray)
    assert placement is explicit_target
    assert placements == [explicit_target]


def test_runtime_device_put_keeps_default_placement_without_runtime_device(monkeypatch):
    """Non-JAX policy remains on the unqualified JAX placement path."""
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: None)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.runtime_device_put([1, 2, 3])

    assert isinstance(array, np.ndarray)
    assert placement is None
    assert placements == [None]


def test_explicit_device_array_preserves_requested_float_dtype(monkeypatch):
    """Explicit FP32 placement must not be rewritten by runtime FP64 policy."""
    runtime_device = jax.devices()[0]
    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: runtime_device)
    invalidate_backend_cache()
    set_backend("jax_cpu_parity", configure_runtime=False)

    array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float32)

    assert array.dtype == jnp.float32


def test_mixed_compute_dtype_does_not_change_runtime_dtype(monkeypatch):
    monkeypatch.delenv("SIMSOPT_MIXED_PRECISION", raising=False)
    invalidate_backend_cache()
    set_backend("jax_cpu_parity", precision="mixed", configure_runtime=False)

    assert dtypes.compute_np_dtype() == np.dtype(np.float32)
    assert dtypes.compute_dtype() == jnp.float32
    assert dtypes.runtime_np_dtype() == np.dtype(np.float64)


def test_as_compute_array_uses_mixed_dtype_without_rewriting_runtime(monkeypatch):
    runtime_device = jax.devices()[0]
    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", lambda: runtime_device)
    invalidate_backend_cache()
    set_backend("jax_cpu_parity", precision="mixed", configure_runtime=False)

    proposal = dtypes.as_compute_array([1.0, 2.0])
    certificate = dtypes.as_compute_array([1.0, 2.0], dtype=jnp.float64)

    assert proposal.dtype == jnp.float32
    assert certificate.dtype == jnp.float64
    assert dtypes.runtime_np_dtype() == np.dtype(np.float64)
