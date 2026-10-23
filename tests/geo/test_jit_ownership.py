"""Native-input ownership, JAX transformation compatibility, and device routing."""

import unittest
from collections import OrderedDict, UserDict, defaultdict, namedtuple
from contextlib import nullcontext
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import simsopt.geo.jit as jit_module
from simsopt.geo.config import parameters
from simsopt.geo.jit import jit


def aligned_array(size):
    """Pinned CPU/JAX fixture aligned to expose a possible zero-copy input path."""
    storage = np.empty(size + 8, dtype=np.float64)
    offset = (-storage.ctypes.data % 64) // storage.itemsize
    values = storage[offset : offset + size]
    values[:] = np.arange(1, size + 1, dtype=np.float64)
    return values


Arrays = namedtuple("Arrays", ["first", "rest"])


class HashableArray(np.ndarray):
    __eq__ = object.__eq__

    # A deliberately hashable ndarray fixture exercises JAX static arguments.
    def __hash__(self) -> int:
        return object.__hash__(self)


def _static_array():
    constant = np.array([3.0]).view(HashableArray)
    constant.flags.writeable = False
    return constant


class IndexSelector:
    def __index__(self):
        return 1


class FrozenDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        raise TypeError("FrozenDictionary does not allow item assignment")


class TransformDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        super().__setitem__(key, 2 * value)


@jax.tree_util.register_pytree_node_class
class RegisteredArrays:
    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0])


class JitOwnershipTests(unittest.TestCase):
    def test_no_jit_snapshots_nested_inputs_and_retains_plain_execution(self):
        def record(x, *, nested):
            return x, nested

        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit_module.jit(record)
        x, y = np.ones(3), np.ones(4)
        with mock.patch.object(jit_module, "native_jax_device") as device:
            held, nested = wrapped(x, nested={"arrays": [y]})
        x[:] = 4
        y[:] = 7
        np.testing.assert_array_equal(held, np.ones(3))
        np.testing.assert_array_equal(nested["arrays"][0], np.ones(4))
        device.assert_not_called()
        self.assertEqual(wrapped.__name__, record.__name__)

    def test_compiled_dispatch_receives_owned_inputs_under_native_device_context(self):
        device_sentinel = object()
        static_sentinel = object()
        x = np.ones(4)
        fake_compile = mock.Mock(side_effect=lambda fun, **kwargs: fun)
        context = mock.Mock(side_effect=lambda device: nullcontext())
        with mock.patch.dict(parameters, {"jit": True}), mock.patch.object(
            jit_module, "jaxjit", fake_compile
        ), mock.patch.object(
            jit_module, "native_jax_device", return_value=device_sentinel
        ), mock.patch.object(jit_module.jax, "default_device", context):
            wrapped = jit_module.jit(lambda a, marker: (a, marker), static_argnums=(1,))
            held, marker = wrapped(x, static_sentinel)
        x[:] = 5
        np.testing.assert_array_equal(held, np.ones(4))
        self.assertIs(marker, static_sentinel)
        context.assert_called_once_with(device_sentinel)
        self.assertEqual(fake_compile.call_args.kwargs, {"static_argnums": (1,)})

    def test_real_jax_outputs_keep_call_time_values_after_input_mutation(self):
        """Even ready identity outputs cannot retain a mutable native input alias."""
        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(
                parameters, {"jit": use_jit}
            ):
                wrapped = jit_module.jit(lambda values: jnp.asarray(values))
                values = aligned_array(16)
                expected = values.copy()
                held = wrapped(values).block_until_ready()
                values[:] = -7.0
                np.testing.assert_array_equal(held, expected)
                self.assertEqual(held.devices(), {jit_module.native_jax_device()})

    def test_real_jax_nested_and_keyword_arrays_keep_owned_values(self):
        def identities(bundle, *, other):
            return jnp.asarray(bundle[0]["arrays"][0]), jnp.asarray(other["value"])

        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(
                parameters, {"jit": use_jit}
            ):
                wrapped = jit_module.jit(identities)
                x, y = aligned_array(8), aligned_array(16)
                expected_x, expected_y = x.copy(), y.copy()
                held_x, held_y = wrapped([{"arrays": (x,)}], other={"value": y})
                held_x.block_until_ready()
                held_y.block_until_ready()
                x[:] = -3.0
                y[:] = -4.0
                np.testing.assert_array_equal(held_x, expected_x)
                np.testing.assert_array_equal(held_y, expected_y)

    def test_real_jax_static_arguments_and_wraps_metadata_are_preserved(self):
        def powers(values, order, *, scale):
            """Scaled powers with positional and keyword static parameters."""
            return scale * jnp.sum(values**order)

        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit_module.jit(
                powers, static_argnums=(1,), static_argnames=("scale",)
            )
        values = np.array([1.0, 2.0, 3.0])
        result = wrapped(values, 3, scale=0.5)
        self.assertEqual(float(result), 18.0)
        self.assertEqual(result.devices(), {jit_module.native_jax_device()})
        self.assertIs(wrapped.__wrapped__, powers)
        self.assertEqual(wrapped.__name__, powers.__name__)
        self.assertEqual(wrapped.__doc__, powers.__doc__)

    def test_tracers_grad_hessian_nested_jit_and_vmap_preserve_derivatives(self):
        values = np.linspace(0.25, 1.0, 4)
        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(
                parameters, {"jit": use_jit}
            ):
                wrapped = jit_module.jit(lambda x: jnp.sum(x**3))
                np.testing.assert_allclose(
                    jax.grad(wrapped)(values), 3 * values**2, rtol=1e-14, atol=0
                )
                np.testing.assert_allclose(
                    jax.jacfwd(jax.grad(wrapped))(values),
                    np.diag(6 * values),
                    rtol=1e-14,
                    atol=0,
                )
                nested = jax.jit(jax.grad(wrapped))
                np.testing.assert_allclose(
                    nested(values), 3 * values**2, rtol=1e-14, atol=0
                )
                batch = np.stack([values, 2 * values])
                np.testing.assert_allclose(
                    jax.vmap(wrapped)(batch),
                    np.sum(batch**3, axis=1),
                    rtol=1e-14,
                    atol=0,
                )

    def test_plain_builtin_mappings_preserve_key_order_and_factory(self):
        mappings: list[dict[object, np.ndarray | list[float]]] = [
            {1: np.array([2.0]), "a": np.array([3.0])},
            {"z": np.array([2.0]), "a": np.array([3.0])},
            OrderedDict([("z", np.array([2.0])), ("a", np.array([3.0]))]),
            defaultdict(list, {1: np.array([2.0]), "a": np.array([3.0])}),
            defaultdict(list, {"z": np.array([2.0]), "a": np.array([3.0])}),
        ]
        with mock.patch.dict(parameters, {"jit": False}):
            identity = jit(lambda values: values)
        for mapping in mappings:
            with self.subTest(kind=type(mapping), keys=tuple(mapping)):
                nested = Arrays([mapping], None)
                copied = identity(nested).first[0]
                self.assertIs(type(copied), type(mapping))
                self.assertEqual(tuple(copied), tuple(mapping))
                for key in mapping:
                    self.assertFalse(np.shares_memory(copied[key], mapping[key]))
                    np.testing.assert_array_equal(copied[key], mapping[key])
        self.assertIs(identity(mappings[-1]).default_factory, list)

    def test_scalar_array_static_selector_uses_index_protocol(self):
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(
                lambda values, offset: values + offset,
                static_argnums=np.array(1, dtype=np.int64),
            )
        np.testing.assert_array_equal(wrapped(jnp.array([3.0]), 2), [5.0])

    def test_custom_static_selector_uses_index_protocol(self):
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(
                lambda values, offset: values + offset, static_argnums=IndexSelector()
            )
        np.testing.assert_array_equal(wrapped(jnp.array([3.0]), 2), [5.0])

    def test_getattr_retains_explicit_static_without_signature(self):
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(getattr, static_argnums=1)
        self.assertEqual(
            tuple(int(item) for item in wrapped(jnp.array([3.0, 4.0]), "shape")), (2,)
        )

    def test_min_retains_explicit_statics_without_signature(self):
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(min, static_argnums=(0, 1))
        self.assertEqual(int(wrapped(3, 2)), 2)

    def test_frozen_dictionary_is_an_opaque_leaf_without_hooks(self):
        incoming = FrozenDictionary({"entry": np.array([3.0])})
        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit(lambda values: values)
        self.assertIs(wrapped(incoming), incoming)
        np.testing.assert_array_equal(wrapped(incoming)["entry"], [3.0])
        self.assertEqual(incoming.setter_calls, 0)

    def test_transform_dictionary_is_an_opaque_leaf_without_hooks(self):
        incoming = TransformDictionary({"entry": np.array([3.0])})
        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit(lambda values: values)
        self.assertIs(wrapped(incoming), incoming)
        np.testing.assert_array_equal(wrapped(incoming)["entry"], [3.0])
        self.assertEqual(incoming.setter_calls, 0)

    def test_opaque_userdict_identity_and_contents_are_retained(self):
        incoming = UserDict({"entry": np.array([3.0])})
        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit(lambda values: values)
        self.assertIs(wrapped(incoming), incoming)
        self.assertIs(wrapped(incoming)["entry"], incoming["entry"])

    def test_registered_custom_pytree_snapshots_dynamic_numpy_leaves(self):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit), mock.patch.dict(
                parameters, {"jit": use_jit}
            ):
                wrapped = jit(lambda container: jnp.asarray(container.values))
                incoming = RegisteredArrays(aligned_array(16))
                expected = incoming.values.copy()
                held = wrapped(incoming).block_until_ready()
                incoming.values[:] = -7
                np.testing.assert_array_equal(held, expected)

    def test_matrix_semantics_and_storage_with_jit_off(self):
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        matrix = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        copied, result = square(matrix)
        self.assertIsInstance(copied, np.matrix)
        self.assertIsInstance(result, np.matrix)
        self.assertFalse(np.shares_memory(copied, matrix))
        matrix[:] = -7
        np.testing.assert_array_equal(copied, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, [[7.0, 10.0], [15.0, 22.0]])

    def test_mask_semantics_and_storage_with_jit_off(self):
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        masked = np.ma.array([2.0, 9.0], mask=[False, True])
        copied, result = square(masked)
        self.assertIsInstance(copied, np.ma.MaskedArray)
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertFalse(np.shares_memory(copied.data, masked.data))
        self.assertFalse(np.shares_memory(copied.mask, masked.mask))
        masked.data[:] = -7
        np.ma.getmaskarray(masked)[:] = False
        np.testing.assert_array_equal(copied.data, [2.0, 9.0])
        np.testing.assert_array_equal(copied.mask, [False, True])
        np.testing.assert_array_equal(result.mask, [False, True])
        self.assertEqual(result.sum(), 4.0)

    def test_static_array_identity_and_compilation_cache(self):
        """Static NumPy objects retain identity for tracing and repeated cache hits."""
        for options in (
            {"static_argnums": 1},
            {"static_argnums": (1,)},
            {"static_argnums": np.int64(1)},
            {"static_argnames": "constant"},
        ):
            for keyword in (False, True):
                with self.subTest(options=options, keyword=keyword):
                    traced_constants = []

                    def add_static(value, constant):
                        traced_constants.append(constant)
                        return value + constant[0]

                    constant = _static_array()
                    with mock.patch.dict(parameters, {"jit": True}):
                        wrapped = jit(add_static, **options)
                    for _ in range(3):
                        result = (
                            wrapped(2.0, constant=constant)
                            if keyword
                            else wrapped(2.0, constant)
                        )
                        self.assertEqual(float(result), 5.0)
                    self.assertEqual(len(traced_constants), 1)
                    self.assertIs(traced_constants[0], constant)

    def test_explicit_static_selectors_keep_actual_call_positioning(self):
        """Explicit selectors do not infer the other selector or bind keyword calls."""
        for options, keyword, is_static in (
            ({"static_argnums": 1, "static_argnames": ()}, False, True),
            ({"static_argnums": 1, "static_argnames": ()}, True, False),
            ({"static_argnums": (), "static_argnames": "constant"}, False, False),
            ({"static_argnums": (), "static_argnames": "constant"}, True, True),
            ({"static_argnums": -1}, False, True),
            ({"static_argnums": -1}, True, False),
        ):
            with self.subTest(options=options, keyword=keyword):
                traced_constants = []

                def add_static(value, constant):
                    traced_constants.append(constant)
                    return value + constant[0]

                constant = _static_array()
                with mock.patch.dict(parameters, {"jit": True}):
                    wrapped = jit(add_static, **options)
                for _ in range(3):
                    result = (
                        wrapped(2.0, constant=constant)
                        if keyword
                        else wrapped(2.0, constant)
                    )
                    self.assertEqual(float(result), 5.0)
                self.assertEqual(len(traced_constants), 1)
                self.assertEqual(traced_constants[0] is constant, is_static)

    def test_negative_static_index_in_variadic_call(self):
        traced_constants = []

        def add_last(value, *constants):
            traced_constants.append(constants[-1])
            return value + constants[0] + constants[-1][0]

        constant = _static_array()
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(add_last, static_argnums=-1)
        for _ in range(3):
            self.assertEqual(float(wrapped(2.0, 4.0, constant)), 9.0)
        self.assertEqual(len(traced_constants), 1)
        self.assertIs(traced_constants[0], constant)

    def test_static_selector_iterators_are_consumed_once(self):
        for options in (
            {"static_argnums": iter((1,))},
            {"static_argnames": iter(("constant",))},
        ):
            with self.subTest(option=next(iter(options))):
                traced_constants = []

                def add_static(value, constant):
                    traced_constants.append(constant)
                    return value + constant[0]

                constant = _static_array()
                with mock.patch.dict(parameters, {"jit": True}):
                    wrapped = jit(add_static, **options)
                for _ in range(3):
                    self.assertEqual(float(wrapped(2.0, constant)), 5.0)
                self.assertEqual(len(traced_constants), 1)
                self.assertIs(traced_constants[0], constant)

    def test_keyword_only_static_subtree_is_not_copied(self):
        traced_options = []

        def add_static(value, *, options):
            traced_options.append(options)
            return value + options[0][0]

        options = (_static_array(),)
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(add_static, static_argnames=("options",))
        for _ in range(3):
            self.assertEqual(float(wrapped(2.0, options=options)), 5.0)
        self.assertEqual(len(traced_options), 1)
        self.assertIs(traced_options[0], options)

    def test_jax_input_donation_is_preserved(self):
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(lambda value: value + 1, donate_argnums=0)
        value = jax.device_put(np.ones(3), jax.devices("cpu")[0])
        np.testing.assert_array_equal(wrapped(value), np.full(3, 2.0))
        self.assertTrue(value.is_deleted())


if __name__ == "__main__":
    unittest.main()
