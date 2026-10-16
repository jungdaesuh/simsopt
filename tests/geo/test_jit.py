import unittest
from collections import namedtuple
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from simsopt.geo.config import parameters
from simsopt.geo.jit import jit


def _aligned(n):
    """A 64-byte aligned array that the CPU backend can read without copying."""
    storage = np.empty(n + 8)
    offset = (-storage.ctypes.data % 64) // storage.itemsize
    array = storage[offset:offset + n]
    array[:] = 1.0
    return array


@jax.jit
def _gate(value, matrix):
    product = lax.fori_loop(0, 8, lambda _, y: (y @ matrix) / matrix.shape[0], matrix)
    return jnp.where(product[0, 0] > 0, value, -value)


def _sum_when_ready(x, ready):
    return jnp.sum(jnp.where(ready[0] > 0, x, -x))


Arrays = namedtuple('Arrays', ['first', 'rest'])


class HashableArray(np.ndarray):
    __eq__ = object.__eq__

    def __hash__(self) -> int:
        return object.__hash__(self)


def _static_array():
    constant = np.array([3.0]).view(HashableArray)
    constant.flags.writeable = False
    return constant


class JitTests(unittest.TestCase):
    def test_numpy_argument_is_copied_before_dispatch(self):
        """Only a still-pending dependency proves the input was overwritten before use."""
        with mock.patch.dict(parameters, {'jit': True}):
            sum_when_ready = jit(_sum_when_ready)
        x = _aligned(1024)
        matrix = jnp.ones((2048, 2048), dtype=jnp.float64)
        ready = _gate(jnp.ones(1), matrix).block_until_ready()
        if not hasattr(ready, 'is_ready'):
            self.skipTest('this JAX version cannot report asynchronous readiness')
        sum_when_ready(x, ready).block_until_ready()
        for _ in range(5):
            x[:] = 1.0
            gate = _gate(ready, matrix)
            result = sum_when_ready(x, gate)
            x[:] = -7.0
            if not gate.is_ready():
                self.assertEqual(float(result), 1024.0)
                return
            result.block_until_ready()
        self.skipTest('the gate finished before the argument changed in every trial')

    def test_numpy_leaves_are_owned_with_jit_off(self):
        """Returned NumPy leaves cannot alias native caches, including nested kwargs."""
        with mock.patch.dict(parameters, {'jit': False}):
            identity = jit(lambda x, *, nested: (x, nested))
        x, y = np.ones(3), np.ones(8)[::2]
        nested = {'arrays': Arrays(y, [None, 'static'])}
        result_x, result_nested = identity(x, nested=nested)
        x[:] = -7
        y[:] = -9
        np.testing.assert_array_equal(result_x, np.ones(3))
        np.testing.assert_array_equal(result_nested['arrays'].first, np.ones(4))
        self.assertFalse(np.shares_memory(result_x, x))
        self.assertFalse(np.shares_memory(result_nested['arrays'].first, y))
        self.assertIsInstance(result_nested['arrays'], Arrays)
        self.assertEqual(result_nested['arrays'].rest, [None, 'static'])

    def test_jax_leaves_are_preserved_with_jit_off(self):
        """Ownership of NumPy inputs does not copy explicitly committed JAX inputs."""
        x = jax.device_put(np.ones(3), jax.devices('cpu')[0])
        with mock.patch.dict(parameters, {'jit': False}):
            identity = jit(lambda *, nested: nested)
        self.assertIs(identity(nested={'arrays': [x]})['arrays'][0], x)

    def test_nested_inputs_and_static_arguments_with_jit_on(self):
        def scaled_sum(nested, scale):
            return scale * (jnp.sum(nested['arrays'].first)
                            + jnp.sum(nested['arrays'].rest[0]))

        with mock.patch.dict(parameters, {'jit': True}):
            wrapped = jit(scaled_sum, static_argnums=1)
        nested = {'arrays': Arrays(np.ones(3), [np.ones(4)])}
        self.assertEqual(float(wrapped(nested, 2)), 14.0)
        self.assertIs(wrapped.__wrapped__, scaled_sum)

    def test_matrix_semantics_and_storage_with_jit_off(self):
        with mock.patch.dict(parameters, {'jit': False}):
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
        with mock.patch.dict(parameters, {'jit': False}):
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
        for options in ({'static_argnums': 1}, {'static_argnums': (1,)},
                        {'static_argnums': np.int64(1)},
                        {'static_argnames': 'constant'}):
            for keyword in (False, True):
                with self.subTest(options=options, keyword=keyword):
                    traced_constants = []

                    def add_static(value, constant):
                        traced_constants.append(constant)
                        return value + constant[0]

                    constant = _static_array()
                    with mock.patch.dict(parameters, {'jit': True}):
                        wrapped = jit(add_static, **options)
                    for _ in range(3):
                        result = (wrapped(2.0, constant=constant) if keyword
                                  else wrapped(2.0, constant))
                        self.assertEqual(float(result), 5.0)
                    self.assertEqual(len(traced_constants), 1)
                    self.assertIs(traced_constants[0], constant)

    def test_explicit_static_selectors_keep_actual_call_positioning(self):
        """Explicit selectors do not infer the other selector or bind keyword calls."""
        for options, keyword, is_static in (
            ({'static_argnums': 1, 'static_argnames': ()}, False, True),
            ({'static_argnums': 1, 'static_argnames': ()}, True, False),
            ({'static_argnums': (), 'static_argnames': 'constant'}, False, False),
            ({'static_argnums': (), 'static_argnames': 'constant'}, True, True),
            ({'static_argnums': -1}, False, True),
            ({'static_argnums': -1}, True, False),
        ):
            with self.subTest(options=options, keyword=keyword):
                traced_constants = []

                def add_static(value, constant):
                    traced_constants.append(constant)
                    return value + constant[0]

                constant = _static_array()
                with mock.patch.dict(parameters, {'jit': True}):
                    wrapped = jit(add_static, **options)
                for _ in range(3):
                    result = (wrapped(2.0, constant=constant) if keyword
                              else wrapped(2.0, constant))
                    self.assertEqual(float(result), 5.0)
                self.assertEqual(len(traced_constants), 1)
                self.assertEqual(traced_constants[0] is constant, is_static)

    def test_negative_static_index_in_variadic_call(self):
        traced_constants = []

        def add_last(value, *constants):
            traced_constants.append(constants[-1])
            return value + constants[0] + constants[-1][0]

        constant = _static_array()
        with mock.patch.dict(parameters, {'jit': True}):
            wrapped = jit(add_last, static_argnums=-1)
        for _ in range(3):
            self.assertEqual(float(wrapped(2.0, 4.0, constant)), 9.0)
        self.assertEqual(len(traced_constants), 1)
        self.assertIs(traced_constants[0], constant)

    def test_static_selector_iterators_are_consumed_once(self):
        for options in ({'static_argnums': iter((1,))},
                        {'static_argnames': iter(('constant',))}):
            with self.subTest(option=next(iter(options))):
                traced_constants = []

                def add_static(value, constant):
                    traced_constants.append(constant)
                    return value + constant[0]

                constant = _static_array()
                with mock.patch.dict(parameters, {'jit': True}):
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
        with mock.patch.dict(parameters, {'jit': True}):
            wrapped = jit(add_static, static_argnames=('options',))
        for _ in range(3):
            self.assertEqual(float(wrapped(2.0, options=options)), 5.0)
        self.assertEqual(len(traced_options), 1)
        self.assertIs(traced_options[0], options)

    def test_jax_input_donation_is_preserved(self):
        with mock.patch.dict(parameters, {'jit': True}):
            wrapped = jit(lambda value: value + 1, donate_argnums=0)
        value = jax.device_put(np.ones(3), jax.devices('cpu')[0])
        np.testing.assert_array_equal(wrapped(value), np.full(3, 2.0))
        self.assertTrue(value.is_deleted())


if __name__ == '__main__':
    unittest.main()
