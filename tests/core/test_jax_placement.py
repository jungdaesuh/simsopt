"""Who owns the host buffers JAX reads, and where and in which dtype JAX values land.

1. ``simsopt_jax.backend.jit.jit`` (``CopyingJit``) makes static the arguments
   ``jax.jit`` does, copies the NumPy leaves of the dynamic ones, and leaves
   placement to JAX.
2. The placement helpers of ``simsopt_jax.backend.dtypes`` keep float64, follow
   a committed reference, leave values nobody placed uncommitted, and make
   cross-backend moves explicit.
3. The host boundary owners place and read back values under the strict
   transfer guard, keeping pytree structure and dtypes.
4. Placement, specs, kernels and the field adapter snapshot caller NumPy
   buffers before deferred execution reads them.
5. Captured extraction specs are host-resident, and grouped coil-set specs keep
   supplied specs and own tuple inputs.
"""

from __future__ import annotations

import os
from functools import partial
from typing import cast
from unittest import mock

import numpy as np
from unittest_jax_support import (
    JAX_IMPORT_ERROR,
    JaxTestCase,
    aligned_host_copy,
    optional_jax_imports,
    require_cuda,
    run_fresh_process,
)

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import dtypes, set_backend
    from simsopt_jax.backend import jit as jit_module
    from simsopt_jax.backend.dtypes import (
        as_jax_array,
        as_jax_float64,
        explicit_device_array,
        runtime_device_put,
        runtime_device_put_tree,
    )
    from simsopt_jax.backend.jit import CopyingJit, jit
    from simsopt_jax.backend.runtime import (
        block_until_ready,
        disallow_host_transfers,
        host_array,
        host_tree,
        host_tree_after_ready,
        snapshot_host_tree,
    )
    from simsopt_jax.core import _device_scalars
    from simsopt_jax.core import biotsavart as kernels
    from simsopt_jax.core import specs
    from simsopt_jax.core._device_scalars import staged_like
    from simsopt_jax.core.field import (
        coil_set_spec_from_dof_extraction_spec,
        group_biot_savart_B_vjp,
    )
    from simsopt_jax.core.specs import (
        CoilDofExtractionSpec,
        host_resident_spec,
        make_coil_dof_extraction_spec,
        make_coil_group_spec,
        make_coil_set_dof_extraction_spec,
        make_curve_xyzfourier_spec,
        make_grouped_coil_set_spec,
        make_optimizable_dof_map_spec,
    )
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.field import biotsavart_backend as adapter
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

    # Native simsopt.geo imports jax too.
    from simsopt.field import Coil, Current
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.geo.curvehelical import CurveHelical
    from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries


# ── CopyingJit ────────────────────────────────────────────────────────


class _Table:
    """Pytree holding a NumPy array; hashes by identity, so it can be static."""

    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


if JAX_IMPORT_ERROR is None:
    jax.tree_util.register_pytree_node_class(_Table)


def _scaled(received: list, x, table):
    """``x * sum(table.values)``, appending ``table`` to ``received``."""
    received.append(table)
    return x * jnp.sum(jnp.asarray(table.values))


def _table_probe(kind: str):
    """``_scaled`` with the parameter kinds ``kind`` names, and the tables it receives."""
    received = []
    probes = {
        "positional-only x": lambda x, /, table: _scaled(received, x, table),
        "keyword-only table": lambda x, *, table: _scaled(received, x, table),
        "positional-or-keyword": lambda x, table: _scaled(received, x, table),
        "defaulted last": lambda x, table, label="": _scaled(received, x, table),
        "*args": lambda x, *tables: _scaled(received, x, tables[-1]),
    }
    return received, probes[kind]


def _recording_copies(copies: list):
    """Patch ``snapshot_host_tree`` to append each ``(value, copy)`` pair to ``copies``."""

    def record(value):
        copies.append((value, snapshot_host_tree(value)))
        return copies[-1][1]

    return mock.patch.object(jit_module, "snapshot_host_tree", record)


class TestCopyingJit(JaxTestCase):
    """``jit`` copies the NumPy leaves of dynamic arguments, passes static
    arguments and JAX arrays by identity, forwards the compiled function's
    attributes, and leaves placement to JAX."""

    def test_static_selectors_match_raw_jit(self):
        """Every selector form makes the arguments static that ``jax.jit`` does:
        static ones, even pytrees of NumPy arrays, pass by identity and never
        recompile, and dynamic NumPy arguments are copied."""
        table = _Table(np.arange(3.0))
        positional = lambda f, x: f(x, table)
        keyword = lambda f, x: f(x, table=table)
        # (name, probe kind, compile with the selector, call); each compile
        # builds a fresh iterator.
        cases = (
            (
                "argnums, positional",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnums=1),
                positional,
            ),
            (
                "negative argnums",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnums=-1),
                positional,
            ),
            # -1 names ``label``, which this call leaves out, so ``table`` is dynamic.
            (
                "negative argnums",
                "defaulted last",
                lambda j, fun: j(fun, static_argnums=-1),
                positional,
            ),
            (
                "negative argnums",
                "*args",
                lambda j, fun: j(fun, static_argnums=-1),
                positional,
            ),
            (
                "negative argnums, keyword",
                "defaulted last",
                lambda j, fun: j(fun, static_argnums=-1),
                lambda f, x: f(x, table, label="static"),
            ),
            (
                "argnums, keyword",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnums=1),
                keyword,
            ),
            (
                "argnames, positional",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnames="table"),
                positional,
            ),
            (
                "argnames, keyword",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnames="table"),
                keyword,
            ),
            (
                "argnames",
                "positional-only x",
                lambda j, fun: j(fun, static_argnames="table"),
                positional,
            ),
            (
                "argnums",
                "positional-only x",
                lambda j, fun: j(fun, static_argnums=1),
                keyword,
            ),
            (
                "argnames",
                "keyword-only table",
                lambda j, fun: j(fun, static_argnames="table"),
                keyword,
            ),
            (
                "iterator argnums",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnums=iter((1,))),
                keyword,
            ),
            (
                "iterator argnames",
                "positional-or-keyword",
                lambda j, fun: j(fun, static_argnames=iter(("table",))),
                positional,
            ),
        )
        for name, kind, compile_with, call in cases:
            with self.subTest(selector=name, parameters=kind):
                raw_received, raw_fun = _table_probe(kind)
                raw = compile_with(jax.jit, raw_fun)  # noqa: TID251 raw jax.jit is the reference
                call(raw, np.ones(2))

                received, fun = _table_probe(kind)
                f = compile_with(jit, fun)
                x = np.ones(2)
                copies = []
                with _recording_copies(copies):
                    for _ in range(3):
                        np.testing.assert_array_equal(call(f, x), np.full(2, 3.0))
                self.assertEqual(len(received), 1)
                if (
                    kind == "positional-or-keyword"
                ):  # every selector here makes ``table`` static
                    self.assertIs(received[0], table)
                self.assertEqual(received[0] is table, raw_received[0] is table)
                self.assertEqual(f._cache_size(), 1)
                copied = [value for value, _ in copies]
                self.assertTrue(any(value is x for value in copied), copied)
                self.assertEqual(
                    any(value is table for value in copied),
                    raw_received[0] is not table,
                )

    def test_every_decorator_form_compiles_with_its_options(self):
        """``@jit``, ``partial(jit, ...)``, ``@jit(...)`` and ``jit(f, ...)`` give the same wrapper."""

        def power(x, n):
            return x**n

        @jit
        def bare(x):
            return x**2

        forms = {
            "bare decorator": (bare, lambda f, x: f(x)),
            "partial": (partial(jit, static_argnames="n")(power), lambda f, x: f(x, 2)),
            "decorator factory": (
                jit(static_argnames="n")(power),
                lambda f, x: f(x, n=2),
            ),
            "call": (jit(power, static_argnums=1), lambda f, x: f(x, 2)),
        }
        x = np.linspace(0.0, 1.0, 4)
        for name, (f, call) in forms.items():
            with self.subTest(form=name):
                self.assertIsInstance(f, CopyingJit)
                np.testing.assert_array_equal(call(f, x), x**2)
        self.assertEqual(bare.__name__, "bare")
        self.assertEqual(bare.lower(x).compile()(x).shape, x.shape)
        self.assertEqual(bare.eval_shape(x).shape, x.shape)
        self.assertEqual(bare.trace(x).jaxpr.in_avals[0].shape, x.shape)
        self.assertEqual(bare._cache_size(), 1)
        bare.clear_cache()
        self.assertEqual(bare._cache_size(), 0)
        np.testing.assert_allclose(jax.grad(lambda y: jnp.sum(bare(y)))(x), 2 * x)
        np.testing.assert_array_equal(jit(lambda y: bare(y) + 1.0)(x), x**2 + 1.0)

        class Scaler:
            factor = 3.0

            @partial(jit, static_argnums=0)
            def scale(self, y):
                return self.factor * y

        np.testing.assert_array_equal(Scaler().scale(x), 3.0 * x)

    def test_jax_array_inputs_are_not_copied(self):
        """Arguments without NumPy leaves skip the copy; NumPy leaves are copied privately."""
        copies = []
        add = jit(lambda a, tree: a + tree["b"][0] * tree["b"][1])
        values = jnp.arange(3.0)
        host = np.arange(3.0)
        with _recording_copies(copies):
            add(values, {"b": [values, values]})
            add(values, tree={"b": (values, 2.0)})
            self.assertEqual(copies, [])
            add(values, {"b": [host, values]})
        (_, first), (_, tree) = copies
        self.assertIs(first, values)
        self.assertIs(tree["b"][1], values)
        self.assertIsNot(tree["b"][0], host)
        np.testing.assert_array_equal(tree["b"][0], host)

    def test_masked_array_is_rejected_like_raw_jit(self):
        """A NumPy masked array keeps its mask through the copy, so JAX rejects it
        as ``jax.jit`` does instead of computing with the masked entries."""
        masked = np.ma.masked_array(np.arange(3.0), mask=[False, True, False])
        with self.assertRaisesRegex(ValueError, "masked arrays are not supported"):
            jit(lambda x: x * 1.0)(masked)

    def test_placement_matches_raw_jit(self):
        """NumPy, uncommitted and committed inputs land where ``jax.jit`` puts them."""
        identity = jit(lambda x: x * 1.0)
        raw = jax.jit(lambda x: x * 1.0)  # noqa: TID251 reference placement of raw jit
        host = np.arange(4.0)
        cpu = jax.devices("cpu")[0]
        inputs = {
            "NumPy": host,
            "uncommitted": jnp.asarray(host),
            "committed CPU": jax.device_put(host, cpu),  # noqa: TID251 committed input under test
        }
        for name, value in inputs.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), raw(value).devices())

    def test_runtime_gpu_receives_numpy_and_uncommitted_inputs(self):
        """With the runtime on the GPU, NumPy and uncommitted inputs run there; committed ones stay."""
        require_cuda("CUDA device required for runtime GPU placement")
        set_backend("jax", device="gpu")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        identity = jit(lambda x: x * 1.0)
        host = np.arange(4.0)
        expected = {
            "NumPy": (host, gpu),
            "uncommitted": (jnp.asarray(host), gpu),
            "committed CPU": (jax.device_put(host, cpu), cpu),  # noqa: TID251 committed input under test
            "committed GPU": (jax.device_put(host, gpu), gpu),  # noqa: TID251 committed input under test
        }
        for name, (value, device) in expected.items():
            with self.subTest(input=name):
                self.assertEqual(identity(value).devices(), {device})


# ── Dtype and reference placement ─────────────────────────────────────

# Two host devices exist only if XLA is told so before it initializes, so the
# multi-device checks run this code in a child interpreter (``python -c``),
# with ``JAX_PLATFORMS``, ``JAX_ENABLE_X64`` and ``XLA_FLAGS`` exported by the
# parent; its argument names the scenario.
_PLACEMENT_CHILD = '''
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from simsopt_jax.backend import dtypes
from simsopt_jax.backend.runtime import disallow_host_transfers, set_backend
from simsopt_jax.core._device_scalars import staged_like


def _placed_leaf(value) -> jax.Array:
    """Place ``{"leaf": value}`` with ``runtime_device_put_tree``; return the leaf."""
    leaf = dtypes.runtime_device_put_tree({"leaf": value})["leaf"]
    assert isinstance(leaf, jax.Array), type(leaf)
    return leaf


def staged_device_array() -> None:
    """``staged_like`` moves a device array held elsewhere onto the reference's device."""
    jax.config.update("jax_enable_x64", True)
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


def unplaced_joins_committed() -> None:
    """Unplaced values join data committed to another device; others are moved."""
    jax.config.update("jax_enable_x64", True)
    first, second = jax.devices("cpu")[:2]
    assert dtypes.get_runtime_jax_device() == first
    elsewhere = jax.device_put(np.arange(3.0), second)
    unplaced = dtypes.runtime_device_put(np.ones(3))
    as_float64 = dtypes.as_jax_float64(np.ones(3))
    assert (unplaced + elsewhere).devices() == {second}
    assert (as_float64 * elsewhere).devices() == {second}
    with jax.default_device(second):
        scoped = dtypes.runtime_device_put(np.ones(3))
    assert scoped.devices() == {second}, scoped.devices()
    moved = dtypes.runtime_device_put(elsewhere)
    assert moved.committed and moved.devices() == {first}, moved.devices()

    # An uncommitted array left on another device by an exited default_device
    # scope is not where the runtime puts values: it is moved there.
    with jax.default_device(second):
        left_behind = jnp.arange(3.0)
    assert not left_behind.committed and left_behind.devices() == {second}
    for placed in (
        dtypes.runtime_device_put(left_behind),
        _placed_leaf(left_behind),
    ):
        assert placed.devices() == {first}, placed.devices()
    # One already there stays uncommitted, like a host value.
    assert not _placed_leaf(jnp.zeros(3)).committed

    # A runtime device JAX would not choose is committed, host values included.
    with mock.patch.object(dtypes, "get_runtime_jax_device", lambda: second):
        for placed in (
            dtypes.runtime_device_put(np.ones(3)),
            _placed_leaf(np.ones(3)),
        ):
            assert placed.committed and placed.devices() == {second}, placed.devices()


def cuda_two_backend() -> None:
    """The GPU and the CPU backend place values in one CUDA process."""
    jax.config.update("jax_enable_x64", True)
    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]
    assert dtypes.get_runtime_jax_device() == gpu
    with jax.default_device(cpu):
        left_behind = jnp.arange(3.0)
        scoped = dtypes.runtime_device_put(np.ones(3))
    assert not left_behind.committed and left_behind.devices() == {cpu}
    assert not scoped.committed and scoped.devices() == {cpu}
    for placed in (
        dtypes.runtime_device_put(left_behind),
        _placed_leaf(left_behind),
    ):
        assert placed.devices() == {gpu}, placed.devices()

    set_backend("jax", device="cpu")
    # JAX's default device changed after set_backend: JAX would not choose the CPU.
    jax.config.update("jax_default_device", None)
    assert dtypes.get_runtime_jax_device() == cpu
    for placed in (
        dtypes.runtime_device_put(np.ones(3)),
        _placed_leaf(np.ones(3)),
    ):
        assert placed.committed and placed.devices() == {cpu}, placed.devices()


def cuda_sharded_cross_backend() -> None:
    """Explicit moves between a GPU device and a two-CPU sharding pass a disallow guard."""
    jax.config.update("jax_enable_x64", True)
    gpu = jax.devices("gpu")[0]
    cpus = jax.devices("cpu")[:2]
    cpu_sharding = NamedSharding(Mesh(np.array(cpus), ("i",)), PartitionSpec("i"))
    values = np.arange(6.0)
    on_gpu = jax.device_put(values, gpu)
    with disallow_host_transfers():
        sharded = dtypes.runtime_device_put_tree({"x": on_gpu}, target=cpu_sharding)
        returned = dtypes.runtime_device_put_tree(sharded, device=gpu)
    assert sharded["x"].sharding == cpu_sharding, sharded["x"].sharding
    assert returned["x"].devices() == {gpu}, returned["x"].devices()
    np.testing.assert_array_equal(np.asarray(sharded["x"]), values)
    np.testing.assert_array_equal(np.asarray(returned["x"]), values)


_SCENARIOS = {
    "staged_device_array": staged_device_array,
    "unplaced_joins_committed": unplaced_joins_committed,
    "cuda_two_backend": cuda_two_backend,
    "cuda_sharded_cross_backend": cuda_sharded_cross_backend,
}
_SCENARIOS[sys.argv[1]]()
'''

# The raw Biot-Savart kernel computes in float64 with nothing but
# ``simsopt_jax`` imported: neither ``simsopt.geo`` nor ``set_backend`` runs in
# this child, so only the ``simsopt_jax`` import can enable x64.
_RAW_KERNEL_FLOAT64_CHILD = """
import numpy as np
from simsopt_jax.core.biotsavart import biot_savart_B

radius, current, quadpoints = 1.0, 1.0e5, 64
phi = 2.0 * np.pi * np.arange(quadpoints) / quadpoints
gamma = radius * np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
# d gamma / d t for the normalized parameter t = phi / (2 pi).
gammadash = 2.0 * np.pi * radius * np.stack(
    (-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1
)
heights = np.array([0.0, 0.3, 1.7])
points = np.stack((np.zeros(3), np.zeros(3), heights), axis=-1)
B = biot_savart_B(points, gamma[None], gammadash[None], np.array([current]))
mu0 = 4.0e-7 * np.pi
B_z = mu0 * current * radius**2 / (2.0 * (radius**2 + heights**2) ** 1.5)
assert B.dtype == np.float64, B.dtype
# A float32 field misses this by about 1e-7 relative.
np.testing.assert_allclose(np.asarray(B)[:, 2], B_z, rtol=1e-13, atol=0.0)
np.testing.assert_allclose(np.asarray(B)[:, :2], 0.0, rtol=0.0, atol=1e-12 * B_z.max())
"""


class TestDtypesPlacement(JaxTestCase):
    """Runtime dtype and reference placement stay consistent under tracing and device scopes."""

    def test_reference_placement_short_circuits_on_tracer(self):
        """A tracer reference is never probed for its placement (the O(jaxpr) walk)."""
        captured: dict[str, object] = {}

        @jit
        def f(x):
            with mock.patch.object(dtypes, "_committed_placement") as committed:
                captured["result"] = dtypes._reference_placement(x)
                captured["probe_calls"] = committed.call_count
            return x

        f(jnp.zeros(3))

        self.assertTrue(captured["result"] is None, 'captured["result"] is None')
        self.assertTrue(captured["probe_calls"] == 0, 'captured["probe_calls"] == 0')

    def test_reference_placement_is_the_bare_device_of_a_committed_array(self):
        """A committed concrete reference is probed and reduced to its device."""
        device = jax.local_devices()[0]
        arr = jax.device_put(np.zeros(3), device)  # noqa: TID251 test places on a chosen device
        with mock.patch.object(
            dtypes,
            "_committed_placement",
            wraps=dtypes._committed_placement,
        ) as committed:
            result = dtypes._reference_placement(arr)

        self.assertTrue(committed.call_count == 1, "committed.call_count == 1")
        self.assertTrue(result is device, "result is device")
        self.assertTrue(
            dtypes._reference_placement(jnp.zeros(3)) is None,
            "dtypes._reference_placement(jnp.zeros(3)) is None",
        )

    def test_runtime_device_put_tree_can_preserve_arrays_and_place_host_leaves(self):
        """Preserving placement retains JAX leaf identity and places host leaves without
        changing structure or dtype."""
        array = jax.device_put(np.ones(3, dtype=np.float64), jax.local_devices()[0])  # noqa: TID251 test places on a chosen device
        value = {"device": array, "host": (np.asarray(2, dtype=np.int32), None)}

        placed = dtypes.runtime_device_put_tree(value, preserve_placement=True)

        self.assertTrue(placed["device"] is array, 'placed["device"] is array')
        self.assertTrue(
            isinstance(placed["host"], tuple), 'isinstance(placed["host"], tuple)'
        )
        self.assertTrue(
            isinstance(placed["host"][0], jax.Array),
            'isinstance(placed["host"][0], jax.Array)',
        )
        self.assertTrue(
            placed["host"][0].dtype == np.int32,
            'placed["host"][0].dtype == np.int32',
        )
        self.assertTrue(placed["host"][1] is None, 'placed["host"][1] is None')
        np.testing.assert_array_equal(placed["host"][0], 2)

    def test_staged_like_tracer_does_not_embed_a_runtime_device_put(self):
        """Scalar staging under JIT computes the expected sum without explicit device
        placement."""
        patches = self.patches

        def unexpected_explicit_placement(*args, **kwargs):
            raise AssertionError("traced literals must remain uncommitted")

        patches.enter_context(
            mock.patch.object(
                _device_scalars, "explicit_device_array", unexpected_explicit_placement
            )
        )

        @jit
        def add_staged_scalar(reference):
            return reference + staged_like(reference, 1.0)

        result = add_staged_scalar(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

        np.testing.assert_array_equal(np.asarray(result), np.asarray((2.0, 3.0)))

    def test_staged_like_tracer_preserves_explicit_integer_dtype(self):
        """Scalar staging under JIT preserves a requested int32 dtype."""

        @jit
        def staged_integer(reference):
            return staged_like(reference, 1, dtype=jnp.int32)

        result = staged_integer(jnp.asarray((1.0, 2.0), dtype=jnp.float64))

        self.assertTrue(result.dtype == jnp.int32, "result.dtype == jnp.int32")
        self.assertTrue(int(np.asarray(result)) == 1, "int(np.asarray(result)) == 1")

    def test_staged_like_places_a_device_array_held_elsewhere_with_the_reference(self):
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
        completed = run_fresh_process(
            "-c", _PLACEMENT_CHILD, "staged_device_array", env=environment, timeout=300
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_reference_placement_handles_tracer_leaf_in_sequence(self):
        """The list/tuple branch skips a tracer leaf and returns None."""
        captured: dict[str, object] = {}

        @jit
        def f(x):
            with mock.patch.object(dtypes, "_committed_placement") as committed:
                captured["result"] = dtypes._reference_placement([x])
                captured["probe_calls"] = committed.call_count
            return x

        f(jnp.zeros(3))

        self.assertTrue(captured["result"] is None, 'captured["result"] is None')
        self.assertTrue(captured["probe_calls"] == 0, 'captured["probe_calls"] == 0')

    def test_runtime_device_put_uses_runtime_device_when_no_target(self):
        """Implicit placement follows the runtime device, not JAX defaults."""
        patches = self.patches
        runtime_device = object()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device)
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3])

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is runtime_device, "placement is runtime_device")
        self.assertTrue(
            placements == [runtime_device], "placements == [runtime_device]"
        )

    def test_runtime_device_put_preserves_explicit_target(self):
        """Explicit target placement still takes precedence."""
        patches = self.patches
        explicit_target = object()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        def _unexpected_runtime_device():
            raise AssertionError("explicit placement must not query runtime device")

        patches.enter_context(
            mock.patch.object(
                dtypes, "get_runtime_jax_device", _unexpected_runtime_device
            )
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3], target=explicit_target)

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is explicit_target, "placement is explicit_target")
        self.assertTrue(
            placements == [explicit_target], "placements == [explicit_target]"
        )

    def test_runtime_device_put_keeps_default_placement_without_runtime_device(self):
        """Without a runtime device (native backend, no platform configured) placement
        stays on the unqualified JAX placement path."""
        patches = self.patches
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: None)
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.runtime_device_put([1, 2, 3])

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is None, "placement is None")
        self.assertTrue(placements == [None], "placements == [None]")

    def test_importing_simsopt_jax_alone_computes_the_raw_kernels_in_float64(self):
        """Without ``simsopt.geo`` or ``set_backend``, the raw B kernel matches the
        analytic on-axis loop field to float64 precision, also with x64 disabled
        in the environment."""
        for x64_setting in (None, "false"):
            with self.subTest(JAX_ENABLE_X64=x64_setting):
                environment = dict(os.environ, JAX_PLATFORMS="cpu")
                environment.pop("JAX_ENABLE_X64", None)
                if x64_setting is not None:
                    environment["JAX_ENABLE_X64"] = x64_setting
                completed = run_fresh_process(
                    "-c", _RAW_KERNEL_FLOAT64_CHILD, env=environment, timeout=300
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_device_array_rejects_reduced_precision_floats(self):
        """An exact-dtype placement accepts float64 as its only floating dtype."""
        patches = self.patches
        runtime_device = jax.devices()[0]
        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: runtime_device)
        )
        set_backend("jax", device="cpu")

        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    TypeError, r"Floating dtype must be one of \('float64',\)"
                ):
                    dtypes.explicit_device_array([1.0, 2.0], dtype=dtype)
        array = dtypes.explicit_device_array([1.0, 2.0], dtype=jnp.float64)
        self.assertEqual(array.dtype, jnp.float64)

    def test_explicit_device_array_preserves_single_device_reference(self):
        """Concrete single-device placement must not fall back to the runtime device.

        The reference's device is used verbatim; the runtime device is never
        consulted. The placement is the bare device rather than the reference's
        placement object: the two are identical for an eager put, but the object
        pins a put staged inside ``jit`` to one device (see
        ``dtypes._array_device``).
        """
        patches = self.patches
        reference = jax.device_put(np.zeros(3), jax.local_devices()[0])  # noqa: TID251 test places on a chosen device
        (reference_device,) = reference.devices()
        placements: list[object | None] = []

        def _device_put(array, placement=None):
            placements.append(placement)
            return array, placement

        def _unexpected_runtime_device():
            raise AssertionError("reference placement must not query runtime device")

        patches.enter_context(
            mock.patch.object(
                dtypes, "get_runtime_jax_device", _unexpected_runtime_device
            )
        )
        patches.enter_context(mock.patch.object(dtypes.jax, "device_put", _device_put))

        array, placement = dtypes.explicit_device_array(
            [1.0, 2.0],
            dtype=jnp.float64,
            reference=reference,
        )

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(placement is reference_device, "placement is reference_device")
        self.assertTrue(
            placements == [reference_device], "placements == [reference_device]"
        )

    def test_unplaced_values_stay_uncommitted_like_jax_leaves_them(self):
        """A value no caller placed is uncommitted on the runtime device, as JAX leaves it.

        Committing it (the old rule) claimed a placement no caller made, so it
        refused every computation with data committed elsewhere. A value placed
        with an uncommitted reference is unplaced too; a committed reference's
        device is still used.
        """
        patches = self.patches
        default_device = jax.local_devices()[0]
        patches.enter_context(
            mock.patch.object(dtypes, "get_runtime_jax_device", lambda: default_device)
        )

        unplaced = dtypes.runtime_device_put(np.ones(3))
        unplaced_tree = dtypes.runtime_device_put_tree({"a": np.ones(3)})["a"]
        as_float64 = dtypes.as_jax_float64(np.ones(3))
        with_uncommitted_explicit_reference = dtypes.explicit_device_array(
            np.ones(3), dtype=jnp.float64, reference=jnp.zeros(3)
        )
        with_committed_reference = dtypes.explicit_device_array(
            np.ones(3),
            dtype=jnp.float64,
            reference=jax.device_put(np.zeros(3), default_device),  # noqa: TID251 test places on a chosen device
        )

        for array in (
            unplaced,
            unplaced_tree,
            as_float64,
            with_uncommitted_explicit_reference,
        ):
            self.assertTrue(
                not cast(jax.Array, array).committed,
                "not cast(jax.Array, array).committed",
            )
            self.assertTrue(
                cast(jax.Array, array).devices() == {default_device},
                "cast(jax.Array, array).devices() == {default_device}",
            )
        self.assertTrue(
            with_committed_reference.committed, "with_committed_reference.committed"
        )
        self.assertTrue(
            with_committed_reference.devices() == {default_device},
            "with_committed_reference.devices() == {default_device}",
        )

    def test_unplaced_values_join_data_committed_to_another_device(self):
        """A constant nobody placed joins data committed to another device.

        The committed-to-the-runtime-device rule refused this combination: an
        array built under a CPU ``jax.default_device`` scope met constants
        committed to the runtime device. A
        ``jax.default_device`` scope is honoured as well; an array committed
        elsewhere, or left uncommitted elsewhere by an exited scope, is moved
        onto the runtime device; and a runtime device JAX would not choose is
        committed.
        """
        environment = dict(os.environ)
        environment.update(
            {
                "JAX_PLATFORMS": "cpu",
                "JAX_ENABLE_X64": "1",
                "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            }
        )
        completed = run_fresh_process(
            "-c",
            _PLACEMENT_CHILD,
            "unplaced_joins_committed",
            env=environment,
            timeout=300,
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_unplaced_values_follow_the_runtime_device_in_a_cuda_process(self):
        """Real two-backend placement: the GPU and the CPU backend in one CUDA process.

        An uncommitted CPU array left by a CPU ``jax.default_device`` scope
        goes to the GPU runtime device after the scope; on the CPU backend,
        once JAX's default device was changed back to the GPU, so that JAX
        would not choose the CPU, host values are committed to the CPU.
        """
        require_cuda("CUDA device required for the two-backend placement check")
        environment = dict(os.environ)
        environment.update({"JAX_PLATFORMS": "cuda,cpu", "JAX_ENABLE_X64": "1"})
        completed = run_fresh_process(
            "-c", _PLACEMENT_CHILD, "cuda_two_backend", env=environment, timeout=300
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_explicit_cross_backend_moves_pass_a_disallow_guard(self):
        """Under ``disallow_host_transfers``, explicit moves of a GPU array and a pytree to the CPU and back keep their values."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        array = np.arange(5.0)
        pair = np.ones((2, 3))
        on_gpu = jax.device_put(array, gpu)  # noqa: TID251 test places on a chosen device
        tree = {"array": on_gpu, "pair": (jax.device_put(pair, gpu), np.arange(3))}  # noqa: TID251 test places on a chosen device
        with disallow_host_transfers():
            array_on_cpu = dtypes.runtime_device_put_tree(on_gpu, device=cpu)
            tree_on_cpu = dtypes.runtime_device_put_tree(tree, device=cpu)
            array_back = dtypes.runtime_device_put_tree(array_on_cpu, device=gpu)
            tree_back = dtypes.runtime_device_put_tree(tree_on_cpu, device=gpu)
        for placed, device in [
            (array_on_cpu, cpu),
            (tree_on_cpu, cpu),
            (array_back, gpu),
            (tree_back, gpu),
        ]:
            for leaf in jax.tree.leaves(placed):
                self.assertEqual(leaf.devices(), {device})
                self.assertTrue(leaf.committed)
        for placed in (array_on_cpu, array_back):
            np.testing.assert_array_equal(np.asarray(placed), array)
        for placed in (tree_on_cpu, tree_back):
            np.testing.assert_array_equal(np.asarray(placed["array"]), array)
            np.testing.assert_array_equal(np.asarray(placed["pair"][0]), pair)
            np.testing.assert_array_equal(np.asarray(placed["pair"][1]), np.arange(3))

    def test_runtime_placement_moves_a_gpu_array_to_the_cpu_backend_under_a_disallow_guard(
        self,
    ):
        """The CPU backend in a CUDA process moves a committed GPU array to the CPU under a disallow guard."""
        require_cuda("CUDA device required for cross-backend placement")
        gpu = jax.devices("gpu")[0]
        cpu = jax.devices("cpu")[0]
        values = np.arange(4.0)
        on_gpu = jax.device_put(values, gpu)  # noqa: TID251 test places on a chosen device
        set_backend("jax", device="cpu")
        with disallow_host_transfers():
            placed = dtypes.runtime_device_put_tree({"x": on_gpu})["x"]
        self.assertEqual(placed.devices(), {cpu})
        np.testing.assert_array_equal(np.asarray(placed), values)

    def test_sharded_cross_backend_moves_pass_a_disallow_guard(self):
        """A GPU array moves to a two-CPU sharding and back under a disallow guard, values unchanged."""
        require_cuda("CUDA device required for cross-backend placement")
        environment = dict(os.environ)
        environment.update(
            {
                "JAX_PLATFORMS": "cuda,cpu",
                "JAX_ENABLE_X64": "1",
                "XLA_FLAGS": " ".join(
                    flag
                    for flag in (
                        environment.get("XLA_FLAGS"),
                        "--xla_force_host_platform_device_count=2",
                    )
                    if flag
                ),
            }
        )
        completed = run_fresh_process(
            "-c",
            _PLACEMENT_CHILD,
            "cuda_sharded_cross_backend",
            env=environment,
            timeout=300,
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_commit_in_place_commits_where_the_array_lives(self):
        """An uncommitted array is committed on its own device with its own dtype."""
        uncommitted = jnp.arange(3.0)

        committed = dtypes.commit_in_place(uncommitted)

        self.assertTrue(not uncommitted.committed, "not uncommitted.committed")
        self.assertTrue(committed.committed, "committed.committed")
        self.assertTrue(
            committed.devices() == uncommitted.devices(),
            "committed.devices() == uncommitted.devices()",
        )
        self.assertTrue(
            committed.dtype == uncommitted.dtype, "committed.dtype == uncommitted.dtype"
        )
        np.testing.assert_array_equal(np.asarray(committed), np.arange(3.0))


# ── Host boundary owners ──────────────────────────────────────────────


class TestHostBoundaryOwners(JaxTestCase):
    """The host boundary owners work under the strict transfer guard and keep pytree structure.

    ``simsopt_jax.backend.dtypes`` places host values and
    ``simsopt_jax.backend.runtime`` reads device values back to the host; code
    under ``disallow_host_transfers()`` moves data only through them.
    """

    def test_placement_owners_place_host_values_under_the_strict_guard(self):
        """Explicit placement owners accept host inputs and preserve reference placement
        under a strict guard."""
        tree = {
            "float": np.asarray([1.0, 2.0], dtype=np.float64),
            "integer": (np.asarray(3, dtype=np.int16),),
        }
        reference = jnp.array(np.zeros(2, dtype=np.float64))

        with disallow_host_transfers():
            placed_tree = runtime_device_put_tree(tree)
            placed_array = explicit_device_array(
                [5.0, 6.0], dtype=jnp.float64, reference=reference
            )

        self.assertTrue(
            isinstance(placed_tree["float"], jax.Array),
            'isinstance(placed_tree["float"], jax.Array)',
        )
        np.testing.assert_array_equal(
            jax.device_get(placed_tree["float"]), tree["float"]
        )
        self.assertTrue(
            jax.device_get(placed_tree["integer"][0]).item() == 3,
            'jax.device_get(placed_tree["integer"][0]).item() == 3',
        )
        self.assertTrue(
            placed_array.dtype == jnp.float64, "placed_array.dtype == jnp.float64"
        )
        self.assertTrue(
            placed_array.devices() == reference.devices(),
            "placed_array.devices() == reference.devices()",
        )
        np.testing.assert_array_equal(jax.device_get(placed_array), [5.0, 6.0])

    def test_host_readers_return_writeable_host_copies_under_the_strict_guard(self):
        """Host readers materialize writable arrays with the requested dtype under a
        strict transfer guard."""
        value = {
            "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            "scalar": (jnp.asarray(3, dtype=jnp.int32),),
        }

        with disallow_host_transfers():
            array = host_array(value["vector"], dtype=np.int64)
            tree = host_tree_after_ready(value)
            converted_trees = (
                host_tree(value, dtype=np.int64),
                host_tree_after_ready(value, dtype=np.int64),
            )

        for converted in converted_trees:
            self.assertEqual(converted.keys(), value.keys())
            self.assertIsInstance(converted["scalar"], tuple)
            for name, observed in (
                ("vector", converted["vector"]),
                ("scalar", converted["scalar"][0]),
            ):
                expected = value[name] if name == "vector" else value[name][0]
                self.assertEqual(observed.shape, expected.shape)
                self.assertEqual(observed.dtype, np.dtype(np.int64))
                self.assertTrue(observed.flags.writeable)
                np.testing.assert_array_equal(observed, jax.device_get(expected))
            converted["vector"][0] = -1.0
            np.testing.assert_array_equal(jax.device_get(value["vector"]), [1.0, 2.0])
        self.assertFalse(
            np.shares_memory(
                converted_trees[0]["vector"],
                converted_trees[1]["vector"],
            )
        )

        self.assertTrue(isinstance(array, np.ndarray), "isinstance(array, np.ndarray)")
        self.assertTrue(array.dtype == np.int64, "array.dtype == np.int64")
        self.assertTrue(array.flags.writeable, "array.flags.writeable")
        np.testing.assert_array_equal(array, [1.0, 2.0])
        self.assertTrue(
            isinstance(tree["vector"], np.ndarray),
            'isinstance(tree["vector"], np.ndarray)',
        )
        self.assertTrue(
            tree["vector"].flags.writeable, 'tree["vector"].flags.writeable'
        )
        self.assertTrue(tree["scalar"][0].item() == 3, 'tree["scalar"][0].item() == 3')

    def test_runtime_device_put_tree_preserves_structure_and_exact_leaf_dtypes(self):
        """Tree placement preserves container structure and exact floating and integer
        leaf dtypes."""
        value = {
            "float": np.asarray([1.0, 2.0], dtype=np.float64),
            "integer": (np.asarray(3, dtype=np.int16),),
        }

        placed = runtime_device_put_tree(value)

        self.assertTrue(placed.keys() == value.keys(), "placed.keys() == value.keys()")
        self.assertTrue(
            placed["float"].dtype == jnp.float64, 'placed["float"].dtype == jnp.float64'
        )
        self.assertTrue(
            placed["integer"][0].dtype == jnp.int16,
            'placed["integer"][0].dtype == jnp.int16',
        )

    def test_readiness_and_host_tree_preserve_pytree_structure(self):
        """Readiness and host materialization preserve pytree structure, values and
        writable host storage."""
        value = {
            "vector": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
            "scalar": (jnp.asarray(3, dtype=jnp.int32),),
        }

        ready = block_until_ready(value)
        host = host_tree_after_ready(ready)

        self.assertTrue(host.keys() == value.keys(), "host.keys() == value.keys()")
        np.testing.assert_array_equal(host["vector"], np.asarray([1.0, 2.0]))
        self.assertTrue(host["scalar"][0].item() == 3, 'host["scalar"][0].item() == 3')
        self.assertTrue(
            host["vector"].flags.writeable, 'host["vector"].flags.writeable'
        )


# ── Host buffer ownership ─────────────────────────────────────────────


def _host_array(shape, *, misaligned=False):
    """Distinct values in aligned storage; misaligned storage tests delayed copies."""
    size = int(np.prod(shape))
    return aligned_host_copy(
        np.arange(size).reshape(shape) / max(size, 1) + 1.0, misaligned=misaligned
    )


def _assert_tree_equal(actual, expected):
    jax.block_until_ready(actual)
    for observed, saved in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(observed, saved)


def _assert_pending(result):
    assert any(not leaf.is_ready() for leaf in jax.tree.leaves(result)), (
        "consumer finished before caller mutation"
    )


class _SpecInputs:
    def __init__(self):
        self.dofs = _host_array((10,))
        self.quadpoints = _host_array((65536,))
        self.samples = _host_array((65536, 3))
        self.rotmat = _host_array((3, 3))

    def mutate(self):
        for source in (self.dofs, self.quadpoints, self.samples, self.rotmat):
            source[:] = -7.0


def _curve_spec(factory, source, dof_count=9):
    return factory(dofs=source.dofs[:dof_count], quadpoints=source.quadpoints, order=1)


def _map_spec(source):
    return specs.make_optimizable_dof_map_spec(
        template_full_dofs=source.dofs[:9],
        owner_segments=(),
        input_mode="full",
        input_start=0,
        input_end=9,
    )


_SPEC_FACTORIES = [
    ("xyz", lambda x: _curve_spec(specs.make_curve_xyzfourier_spec, x)),
    (
        "oriented_xyz",
        lambda x: _curve_spec(specs.make_oriented_curve_xyzfourier_spec, x),
    ),
    ("planar", lambda x: _curve_spec(specs.make_curve_planarfourier_spec, x, 10)),
    (
        "rz",
        lambda x: specs.make_curve_rzfourier_spec(
            dofs=x.dofs[:6], quadpoints=x.quadpoints, order=1, nfp=1, stellsym=False
        ),
    ),
    (
        "helical",
        lambda x: specs.make_curve_helical_spec(
            dofs=x.dofs[:3], quadpoints=x.quadpoints, order=1, m=1, ell=1, R0=1.0, r=0.1
        ),
    ),
    (
        "symmetries",
        lambda x: specs.make_curve_xyzfouriersymmetries_spec(
            dofs=x.dofs[:9],
            quadpoints=x.quadpoints,
            order=1,
            nfp=1,
            stellsym=False,
            ntor=1,
        ),
    ),
    ("dof_map", _map_spec),
    (
        "frame_rotation",
        lambda x: specs.make_frame_rotation_spec(
            dofs=x.dofs[:3], quadpoints=x.quadpoints, order=1, scale=1.0
        ),
    ),
    ("zero_rotation", lambda x: specs.make_zero_rotation_spec(quadpoints=x.quadpoints)),
    ("coil_symmetry", lambda x: specs.make_coil_symmetry_spec(rotmat=x.rotmat)),
    ("field", lambda x: specs.make_field_eval_spec(x.samples)),
    (
        "coil_group",
        lambda x: specs.make_coil_group_spec(
            x.samples[None], x.samples[None], x.dofs[:1], (0,)
        ),
    ),
    (
        "perturbed",
        lambda x: specs.make_curve_perturbed_spec(
            dofs=x.dofs[:9],
            quadpoints=x.quadpoints,
            base_curve=_curve_spec(specs.make_curve_xyzfourier_spec, x),
            base_curve_map=_map_spec(x),
            sample_gamma=x.samples,
            sample_gammadash=x.samples,
            sample_gammadashdash=x.samples,
            sample_gammadashdashdash=x.samples,
            sample_derivatives=3,
        ),
    ),
    (
        "filament",
        lambda x: specs.make_curve_filament_spec(
            dofs=x.dofs[:9],
            quadpoints=x.quadpoints,
            base_curve=_curve_spec(specs.make_curve_xyzfourier_spec, x),
            base_curve_map=_map_spec(x),
            rotation=specs.make_zero_rotation_spec(quadpoints=x.quadpoints),
            rotation_map=_map_spec(x),
            frame_kind="frenet",
            dn=0.1,
            db=0.1,
        ),
    ),
]


_GATE_SIZE = 2896


def make_execution_gate():
    """Queue enough CPU work for the warmed consumer to remain pending."""
    matrix = jnp.ones((_GATE_SIZE, _GATE_SIZE), dtype=jnp.float64)

    @jit
    def gate(value, matrix):
        product = jax.lax.fori_loop(
            0, 4, lambda _, x: (x @ matrix) / _GATE_SIZE, matrix
        )
        return jnp.where(product[0, 0] > 0, value, -value)

    return lambda value: gate(value, matrix)


def _kernel_inputs():
    points = _host_array((32, 3))
    points *= 0.2
    gamma = _host_array((1, 64, 3))
    dash = _host_array((1, 64, 3))
    current = _host_array((1,))
    return points, gamma, dash, current


if JAX_IMPORT_ERROR is None:
    _FORWARD_KERNELS = [
        kernels.biot_savart_B,
        kernels.biot_savart_A,
        kernels.biot_savart_dB_by_dX,
        kernels.biot_savart_dA_by_dX,
        kernels.biot_savart_d2B_by_dXdX,
        kernels.biot_savart_d2A_by_dXdX,
        kernels.biot_savart_B_and_dB,
    ]


def _field():
    curve = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=64)[0]
    return JaxBiotSavart([Coil(curve, Current(1e5))]).set_points(
        _host_array((32, 3)) * 0.2
    )


class TestBufferOwnership(JaxTestCase):
    """Caller mutation cannot change captured JAX field state or pending kernels."""

    def test_placement_snapshots_numpy_before_dispatch(self):
        """Placement snapshots aligned and misaligned NumPy inputs before caller
        mutation."""
        for place in [
            runtime_device_put,
            runtime_device_put_tree,
            lambda x: runtime_device_put_tree(x, preserve_placement=True),
            lambda x: explicit_device_array(x, dtype=np.float64),
            lambda x: as_jax_array(x, dtype=np.float64),
            as_jax_float64,
        ]:
            for misaligned in [False, True]:
                with self.subTest(place=place, misaligned=misaligned), self.case():
                    self._case_placement_snapshots_numpy_before_dispatch(
                        place, misaligned
                    )

    def _case_placement_snapshots_numpy_before_dispatch(self, place, misaligned):
        """Run one row with case-local objects released before runtime cleanup."""
        set_backend("jax", device="cpu")
        source = _host_array((5_000_000,), misaligned=misaligned)
        expected = source.copy()
        result = place(source)
        source[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_snapshot_preserves_device_arrays_and_tracers(self):
        """Snapshotting preserves JAX array identity and traced values while supporting
        mixed host/device pytrees."""
        device = jnp.arange(16, dtype=jnp.float64)
        self.assertTrue(
            snapshot_host_tree(device) is device, "snapshot_host_tree(device) is device"
        )
        tree = snapshot_host_tree({"device": device, "host": np.arange(16)})
        self.assertTrue(tree["device"] is device, 'tree["device"] is device')
        np.testing.assert_array_equal(jit(snapshot_host_tree)(device), device)
        self.assertTrue(
            runtime_device_put_tree(device, preserve_placement=True) is device,
            "runtime_device_put_tree(device, preserve_placement=True) is device",
        )

    def test_mixed_device_host_sequence_snapshots_numpy(self):
        """Mixed host/device sequence conversion snapshots NumPy leaves before deferred
        execution reads them."""
        for convert in [
            lambda x: as_jax_array(x, dtype=np.float64),
            as_jax_float64,
            runtime_device_put,
        ]:
            with self.subTest(convert=convert), self.case():
                self._case_mixed_device_host_sequence_snapshots_numpy(convert)

    def _case_mixed_device_host_sequence_snapshots_numpy(self, convert):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        source = _host_array((65536,))
        expected = source.copy()
        zeros = jnp.zeros(source.shape)
        gate(zeros).block_until_ready()
        convert([zeros, source]).block_until_ready()
        result = convert([gate(zeros), source])
        _assert_pending(result)
        source[:] = -7.0
        np.testing.assert_array_equal(result.block_until_ready()[1], expected)

    def test_spec_factories_own_numpy_leaves(self):
        """Spec factories retain owned NumPy snapshots after every caller input is
        mutated."""
        for case_id_0, factory in _SPEC_FACTORIES:
            with self.subTest(case_id_0=case_id_0, factory=factory), self.case():
                self._case_spec_factories_own_numpy_leaves(factory)

    def _case_spec_factories_own_numpy_leaves(self, factory):
        """Run one row with case-local objects released before runtime cleanup."""
        source = _SpecInputs()
        result = factory(source)
        expected = jax.tree.map(lambda leaf: np.array(leaf, copy=True), result)
        source.mutate()
        _assert_tree_equal(result, expected)

    def test_adapter_curve_capture_owns_native_coefficients(self):
        """Adapter curve capture owns snapshots of native coefficients and quadrature
        nodes."""
        for case_id_0, native_curve in zip(
            ["helical", "symmetries"],
            [
                lambda points: CurveHelical(points, order=1),
                lambda points: CurveXYZFourierSymmetries(
                    points, order=1, nfp=1, stellsym=False
                ),
            ],
            strict=True,
        ):
            with self.subTest(
                case_id_0=case_id_0, native_curve=native_curve
            ), self.case():
                self._case_adapter_curve_capture_owns_native_coefficients(native_curve)

    def _case_adapter_curve_capture_owns_native_coefficients(self, native_curve):
        """Run one row with case-local objects released before runtime cleanup."""
        quadpoints = _host_array((65536,))
        quadpoints[:] -= 1.0
        curve = native_curve(quadpoints)
        coefficients = _host_array((curve.num_dofs(),))
        curve.coefficients = coefficients
        result = curve_spec_from_adapter_curve(curve)
        expected = jax.tree.map(lambda x: np.array(x, copy=True), result)
        coefficients[:] = -7.0
        quadpoints[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_cylindrical_points_own_numpy_before_conversion(self):
        """Cylindrical setters snapshot caller coordinates before deferred conversion to
        Cartesian points."""
        for misaligned in [False, True]:
            with self.subTest(misaligned=misaligned), self.case() as patches:
                self._case_cylindrical_points_own_numpy_before_conversion(
                    misaligned, patches
                )

    def _case_cylindrical_points_own_numpy_before_conversion(self, misaligned, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        source = _host_array((1_666_667, 3), misaligned=misaligned)
        saved = source.copy()
        canonical = adapter._canonical_set_points_cyl
        gate(jnp.asarray(saved)).block_until_ready()
        expected = np.array(
            jax.block_until_ready(
                adapter._cyl_points_to_cart(canonical(jnp.asarray(saved)))
            ),
            copy=True,
        )
        patches.enter_context(
            mock.patch.object(
                adapter,
                "_canonical_set_points_cyl",
                lambda points: canonical(gate(points)),
            )
        )
        field = JaxBiotSavart([]).set_points_cyl(source)
        _assert_pending(field._points_jax)
        source[:] = -7.0
        np.testing.assert_array_equal(
            jax.block_until_ready(field._points_jax), expected
        )

    def test_raw_forward_kernel_owns_numpy_inputs(self):
        """Raw forward kernels snapshot NumPy inputs before deferred device execution and
        caller mutation."""
        for pending_input in ["points", "gamma"]:
            for kernel in _FORWARD_KERNELS:
                with self.subTest(
                    pending_input=pending_input,
                    case_id_1=kernel.__name__,
                    kernel=kernel,
                ), self.case():
                    self._case_raw_forward_kernel_owns_numpy_inputs(
                        pending_input, kernel
                    )

    def _case_raw_forward_kernel_owns_numpy_inputs(self, pending_input, kernel):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        inputs = list(_kernel_inputs())
        expected = jax.tree.map(
            lambda x: np.array(x, copy=True),
            jax.block_until_ready(kernel(*inputs)),
        )
        index = 0 if pending_input == "points" else 1
        gate(jnp.asarray(inputs[index])).block_until_ready()
        inputs[index] = gate(jnp.asarray(inputs[index]))
        result = kernel(*inputs)
        _assert_pending(result)
        for source in inputs:
            if isinstance(source, np.ndarray):
                source[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_raw_pullback_owns_numpy_inputs(self):
        """Raw pullbacks snapshot geometry, current and cotangent NumPy inputs before
        deferred execution."""
        for case_id_0, kernel in zip(
            ["raw", "grouped"],
            [kernels.biot_savart_B_vjp, group_biot_savart_B_vjp],
            strict=True,
        ):
            with self.subTest(case_id_0=case_id_0, kernel=kernel), self.case():
                self._case_raw_pullback_owns_numpy_inputs(kernel)

    def _case_raw_pullback_owns_numpy_inputs(self, kernel):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        points, gamma, dash, current = _kernel_inputs()
        cotangent = _host_array(points.shape)
        expected = jax.tree.map(
            lambda x: np.array(x, copy=True),
            jax.block_until_ready(kernel(points, cotangent, gamma, dash, current)),
        )
        gate(jnp.asarray(points)).block_until_ready()
        pending_points = gate(jnp.asarray(points))
        result = kernel(pending_points, cotangent, gamma, dash, current)
        _assert_pending(result)
        for source in (cotangent, gamma, dash, current):
            source[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_explicit_dofs_are_owned(self):
        """Explicit-DOF reconstruction retains owned snapshots despite caller mutation
        during deferred execution."""
        for entry in [
            "_normalize_explicit_coil_dofs",
            "coil_specs_from_dofs",
            "grouped_coil_arrays_from_dofs",
            "coil_set_spec_from_dofs",
        ]:
            with self.subTest(entry=entry), self.case() as patches:
                self._case_explicit_dofs_are_owned(entry, patches)

    def _case_explicit_dofs_are_owned(self, entry, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        field = _field()
        dofs = np.array(field.x, dtype=np.float64)
        source = _host_array(dofs.shape)
        source[:] = dofs
        call = getattr(field, entry)
        expected = jax.tree.map(
            lambda x: np.array(x, copy=True),
            jax.block_until_ready(call(source)),
        )
        normalize = field._normalize_explicit_coil_dofs
        gate(normalize(source)).block_until_ready()
        patches.enter_context(
            mock.patch.object(
                field,
                "_normalize_explicit_coil_dofs",
                lambda dofs: gate(normalize(dofs)),
            )
        )
        result = call(source)
        # Grouped builders materialize captured curve constants while lowering;
        # the normalization boundary and per-coil payload remain deferred.
        if entry == "coil_specs_from_dofs":
            _assert_pending(result)
        source[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_adapter_pullback_owns_numpy_cotangent(self):
        """Adapter pullbacks snapshot caller cotangents before deferred execution reads
        them."""
        for entry, shape in [
            ("B_pullback_native", (32, 3)),
            ("A_pullback_native", (32, 3)),
            ("dB_by_dX_pullback_native", (32, 3, 3)),
            ("dA_by_dX_pullback_native", (32, 3, 3)),
        ]:
            with self.subTest(entry=entry, shape=shape), self.case():
                self._case_adapter_pullback_owns_numpy_cotangent(entry, shape)

    def _case_adapter_pullback_owns_numpy_cotangent(self, entry, shape):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        field = _field()
        call = getattr(field, entry)
        source = _host_array(shape)
        expected = jax.tree.map(
            lambda x: np.array(x, copy=True),
            jax.block_until_ready(call(source)),
        )
        gate(field._points_jax).block_until_ready()
        field._points_jax = gate(field._points_jax)
        result = call(source)
        _assert_pending(result)
        source[:] = -7.0
        _assert_tree_equal(result, expected)

    def test_group_coil_data_owns_numpy_inputs(self):
        """Grouping stacked or per-coil data owns NumPy snapshots."""
        for stacked in [False, True]:
            with self.subTest(stacked=stacked), self.case() as patches:
                self._case_group_coil_data_owns_numpy_inputs(stacked, patches)

    def _case_group_coil_data_owns_numpy_inputs(self, stacked, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        gate = make_execution_gate()
        _, gamma, dash, current = _kernel_inputs()
        inputs = (
            (gamma, dash, current)
            if stacked
            else ([gamma[0]], [dash[0]], [current[0:1].reshape(())])
        )
        expected = jax.tree.map(
            lambda x: np.array(x, copy=True),
            jax.block_until_ready(kernels.group_coil_data(*inputs)),
        )
        convert = kernels._as_jax_float64
        if stacked:
            axis0 = kernels._axis0_entries
            for entry in (gamma, current):
                gate(convert(entry)).block_until_ready()
            patches.enter_context(
                mock.patch.object(
                    kernels,
                    "_axis0_entries",
                    lambda x: axis0(gate(convert(x))),
                )
            )
        else:
            for entry in (gamma[0], current[0:1].reshape(())):
                gate(convert(entry)).block_until_ready()
            patches.enter_context(
                mock.patch.object(
                    kernels, "_as_jax_float64", lambda x: gate(convert(x))
                )
            )
        result = kernels.group_coil_data(*inputs)
        _assert_pending(tuple(group[:3] for group in result))
        for source in (gamma, dash, current):
            source[:] = -7.0
        _assert_tree_equal(result, expected)


# ── Spec placement and construction ───────────────────────────────────

CURVE_DOF_COUNT = 9
OWNER_DOFS = np.asarray(
    [0.9, 0.1, -0.2, 0.05, 0.8, 0.15, -0.1, 0.02, 0.3, 1.5e5],
    dtype=np.float64,
)
ROTMAT = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _dof_map_spec(*, size: int, start: int):
    return make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(size, dtype=np.float64),
        owner_segments=[(start, start + size, 0, size)],
        input_mode="full",
        input_start=start,
        input_end=start + size,
    )


def _extraction_spec() -> CoilDofExtractionSpec:
    return make_coil_dof_extraction_spec(
        curve=make_curve_xyzfourier_spec(
            dofs=np.zeros(CURVE_DOF_COUNT, dtype=np.float64),
            quadpoints=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            order=1,
        ),
        curve_map=_dof_map_spec(size=CURVE_DOF_COUNT, start=0),
        current_map=_dof_map_spec(size=1, start=CURVE_DOF_COUNT),
        rotmat=ROTMAT,
        scale=-1.0,
    )


def _coil_field_inputs(spec: CoilDofExtractionSpec):
    return coil_set_spec_from_dof_extraction_spec(
        make_coil_set_dof_extraction_spec([spec]),
        jnp.array(OWNER_DOFS),
    ).field_inputs()


class TestCoilDofExtractionSpecPlacement(JaxTestCase):
    """Captured extraction specs must use host-resident constants under a strict transfer guard.

    Argument specs retain device-resident leaves. These tests exercise both kernel
    call patterns without changing the immutable spec constructors.
    """

    def test_the_maker_leaves_the_extraction_spec_device_resident(self):
        """The argument role is the default, so the maker must not host-place."""
        leaves = jax.tree.leaves(_extraction_spec())

        self.assertTrue(leaves, "leaves")
        off_device = [leaf for leaf in leaves if not isinstance(leaf, jax.Array)]
        self.assertTrue(
            not off_device,
            f"make_coil_dof_extraction_spec left {len(off_device)} leaf/leaves off the "
            "device; passing the spec to a jitted program would then be an implicit "
            "host-to-device transfer",
        )

    def test_host_resident_spec_moves_every_array_leaf_to_the_host(self):
        """Nothing may be left behind for lowering to read back."""
        on_host = host_resident_spec(_extraction_spec())

        still_on_device = [
            leaf for leaf in jax.tree.leaves(on_host) if isinstance(leaf, jax.Array)
        ]
        self.assertTrue(
            not still_on_device,
            f"host_resident_spec left {len(still_on_device)} leaf/leaves on a device; "
            "a program capturing them reads every one back to the host at lowering",
        )
        self.assertTrue(
            all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(on_host)),
            "all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(on_host))",
        )
        self.assertTrue(
            jax.tree.structure(on_host) == jax.tree.structure(_extraction_spec()),
            "jax.tree.structure(on_host) == jax.tree.structure(_extraction_spec())",
        )

    def test_host_placement_reconstructs_the_coil_arrays_bit_for_bit(self):
        """Where the frozen template lives may not change one bit of the coils."""
        device_inputs = _coil_field_inputs(_extraction_spec())
        host_inputs = _coil_field_inputs(host_resident_spec(_extraction_spec()))

        self.assertTrue(
            len(host_inputs) == len(device_inputs) == 1,
            "len(host_inputs) == len(device_inputs) == 1",
        )
        for name, host_array, device_array in zip(
            ("gammas", "gammadashs", "currents"),
            host_inputs[0],
            device_inputs[0],
            strict=True,
        ):
            np.testing.assert_array_equal(
                np.asarray(host_array),
                np.asarray(device_array),
                err_msg=f"host-resident template changed the reconstructed {name}",
            )

    def test_a_program_capturing_the_host_resident_spec_compiles_under_the_strict_guard(
        self,
    ):
        """Lowering may not read the captured template back to the host.

        On a single-memory CPU backend the guard cannot observe that read, so this
        is a GPU-effective pin; the leaf-placement test above holds the property
        everywhere.
        """
        spec = make_coil_set_dof_extraction_spec(
            [host_resident_spec(_extraction_spec())]
        )

        def total_gamma(owner_dofs):
            gammas, _gammadashs, _currents = coil_set_spec_from_dof_extraction_spec(
                spec, owner_dofs
            ).field_inputs()[0]
            return gammas.sum()

        owner_dofs = as_jax_float64(OWNER_DOFS)
        with jax.transfer_guard("disallow"):
            total = jit(total_gamma)(owner_dofs)

        self.assertTrue(
            np.isfinite(np.asarray(jax.device_get(total))),
            "np.isfinite(np.asarray(jax.device_get(total)))",
        )


class TestGroupedCoilSetSpec(JaxTestCase):
    """Grouped-coil construction preserves supplied specs and owns tuple inputs."""

    def test_existing_specs_retain_identity_and_order(self):
        """Existing specs, including repeated references, retain identity and order."""
        first = make_coil_group_spec(
            [[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]], [7.0], (5,)
        )
        second = make_coil_group_spec(
            [[[8.0, 9.0, 10.0]]], [[[11.0, 12.0, 13.0]]], [14.0], (2,)
        )

        result = make_grouped_coil_set_spec([first, second, first])

        self.assertEqual(len(result.groups), 3)
        self.assertIs(result.groups[0], first)
        self.assertIs(result.groups[1], second)
        self.assertIs(result.groups[2], first)

    def test_tuple_inputs_are_converted_and_snapshot_host_arrays(self):
        """Tuple conversion retains float64 values after callers mutate host inputs."""
        gammas = aligned_host_copy([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        gammadashs = aligned_host_copy([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
        currents = aligned_host_copy([13.0])

        result = make_grouped_coil_set_spec([(gammas, gammadashs, currents, (4,))])
        gammas[:] = -1.0
        gammadashs[:] = -2.0
        currents[:] = -3.0

        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.coil_indices, (4,))
        np.testing.assert_array_equal(
            group.gammas, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        )
        np.testing.assert_array_equal(
            group.gammadashs, [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )
        np.testing.assert_array_equal(group.currents, [13.0])
        for array in (group.gammas, group.gammadashs, group.currents):
            self.assertEqual(array.dtype, np.dtype("float64"))

    def test_mixed_one_pass_generator_preserves_order(self):
        """A mixed generator is consumed once without dropping or reordering groups."""
        first = make_coil_group_spec(
            [[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]], [7.0], (8,)
        )
        middle = (first.gammas, first.gammadashs, first.currents, (3,))
        last = make_coil_group_spec(
            [[[8.0, 9.0, 10.0]]], [[[11.0, 12.0, 13.0]]], [14.0], (1,)
        )
        groups = (group for group in (first, middle, last))

        result = make_grouped_coil_set_spec(groups)

        self.assertEqual(
            tuple(group.coil_indices for group in result.groups), ((8,), (3,), (1,))
        )
        self.assertIs(result.groups[0], first)
        self.assertIs(result.groups[2], last)
        np.testing.assert_array_equal(result.groups[1].currents, [7.0])
        self.assertEqual(tuple(groups), ())

    def test_empty_iterable_produces_empty_groups(self):
        """An empty iterable constructs an empty immutable groups tuple."""
        result = make_grouped_coil_set_spec(iter(()))

        self.assertEqual(result.groups, ())
