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


if __name__ == '__main__':
    unittest.main()
