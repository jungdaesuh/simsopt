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

from simsopt_jax.backend import dtypes


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
