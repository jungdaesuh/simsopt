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

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import invalidate_backend_cache, set_backend
from simsopt_jax.core import _device_scalars
from simsopt_jax.core._device_scalars import staged_like


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


def test_staged_scalar_uses_replicated_named_sharding() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("device",))
    vector_sharding = NamedSharding(mesh, P("device"))
    reference = jax.device_put(
        np.ones(3, dtype=np.float64),
        vector_sharding,
    )

    with jax.transfer_guard("disallow"):
        scalar = staged_like(reference, 1.0)

    assert scalar.ndim == 0
    assert isinstance(scalar.sharding, NamedSharding)
    assert scalar.sharding.spec == P()


def test_staged_like_tracer_does_not_embed_a_runtime_device_put(monkeypatch):
    def unexpected_explicit_placement(*args, **kwargs):
        raise AssertionError("traced literals must remain uncommitted")

    monkeypatch.setattr(
        _device_scalars,
        "explicit_device_array",
        unexpected_explicit_placement,
    )

    @jax.jit
    def add_staged_scalar(reference):
        return reference + staged_like(reference, 1.0)

    result = add_staged_scalar(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

    np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))


def test_staged_like_tracer_preserves_explicit_integer_dtype():
    @jax.jit
    def staged_integer(reference):
        return staged_like(reference, 1, dtype=jnp.int32)

    result = staged_integer(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

    assert result.dtype == jnp.int32
    assert int(np.asarray(result)) == 1


# Two host devices exist only if XLA is told so before it initializes, so the
# check runs in a child process, bootstrapped onto this checkout's sources the
# way tests/conftest.py is (an installed editable finder would otherwise win
# over PYTHONPATH).
_STAGED_DEVICE_ARRAY_CHILD = """
import sys
from pathlib import Path

repo_root = sys.argv[1]
sys.path.insert(0, repo_root)
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(repo_root) / "src")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from simsopt_jax.core._device_scalars import staged_like

first, second = jax.devices("cpu")[:2]
reference = jax.device_put(np.zeros(2), first)
tolerance = jax.device_put(np.asarray(1.0e-12), second)
budget = jax.device_put(np.asarray(40, dtype=np.int32), second)
with jax.transfer_guard("disallow"):
    staged_tolerance = staged_like(reference, tolerance)
    staged_budget = staged_like(reference, budget, dtype=np.int32)
    total = reference + staged_tolerance
assert staged_tolerance.devices() == {first}, staged_tolerance.devices()
assert staged_budget.devices() == {first}, staged_budget.devices()
assert staged_budget.dtype == np.int32
assert float(staged_tolerance) == 1.0e-12 and int(staged_budget) == 40
assert total.devices() == {first}
"""


def test_staged_like_places_a_device_array_held_elsewhere_with_the_reference():
    """A concrete device array is moved onto the reference's device, explicitly.

    Keeping it where it was (``jnp.asarray``) left, e.g., a solver's tolerance
    on another device than the state it is compared with, so the program that
    combines them was rejected.
    """
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            _STAGED_DEVICE_ARRAY_CHILD,
            str(Path(__file__).resolve().parents[1]),
        ),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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


def test_explicit_device_array_preserves_single_device_reference(monkeypatch):
    """Concrete single-device placement must not fall back to the runtime device.

    The reference's device is used verbatim; the runtime device is never
    consulted. The placement is the bare device rather than the reference's
    concrete ``SingleDeviceSharding``: the two are identical for an eager put,
    but the sharding form pins a put staged inside ``jit`` to one device (see
    ``dtypes._single_device_placement``).
    """
    reference = jnp.zeros(3)
    (reference_device,) = reference.sharding.device_set
    placements: list[object | None] = []

    def _device_put(array, placement=None):
        placements.append(placement)
        return array, placement

    def _unexpected_runtime_device():
        raise AssertionError("reference placement must not query runtime device")

    monkeypatch.setattr(dtypes, "maybe_initialize_distributed_jax", lambda: None)
    monkeypatch.setattr(dtypes, "get_runtime_jax_device", _unexpected_runtime_device)
    monkeypatch.setattr(dtypes.jax, "device_put", _device_put)

    array, placement = dtypes.explicit_device_array(
        [1.0, 2.0],
        dtype=jnp.float32,
        reference=reference,
    )

    assert isinstance(array, np.ndarray)
    assert placement is reference_device
    assert placements == [reference_device]
