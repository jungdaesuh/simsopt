import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from simsopt.geo import jit as jit_module
from simsopt.geo.jit import jit


def _aligned(n):
    """A 64-byte aligned array, which the CPU backend may use without copying."""
    storage = np.empty(n + 8)
    offset = (-storage.ctypes.data % 64) // storage.itemsize
    array = storage[offset:offset + n]
    array[:] = 1.0
    return array


@jax.jit
def _gate(value, matrix):
    product = lax.fori_loop(0, 4, lambda _, y: (y @ matrix) / matrix.shape[0], matrix)
    return jnp.where(product[0, 0] > 0, value, -value)


@jit
def _sum_when_ready(x, ready):
    return jnp.sum(jnp.where(ready[0] > 0, x, -x))


class JitTests(unittest.TestCase):
    def test_numpy_argument_is_copied_before_dispatch(self):
        """Changing a NumPy argument after the call cannot change the result.

        The result depends on a queued gate, so the compiled call cannot read ``x`` until
        the gate finishes, which is after ``x`` is overwritten."""
        x = _aligned(1024)
        matrix = jnp.ones((2048, 2048), dtype=jnp.float32)
        ready = _gate(jnp.ones(1), matrix).block_until_ready()
        _sum_when_ready(x, ready).block_until_ready()
        result = _sum_when_ready(x, _gate(ready, matrix))
        self.assertFalse(result.is_ready(), "the gate finished before the argument changed")
        x[:] = -7.0
        self.assertEqual(float(result), 1024.0)

    def test_numpy_leaves_are_copied_with_jit_off(self):
        """With jit off the wrapped function still receives private copies of NumPy leaves."""
        received = []
        previous = jit_module.parameters['jit']
        jit_module.parameters['jit'] = False
        try:
            record = jit(lambda x, *, nested: received.extend([x, *nested['arrays']]))
        finally:
            jit_module.parameters['jit'] = previous
        x, y = np.ones(3), np.ones(4)
        record(x, nested={'arrays': [y]})
        self.assertFalse(np.shares_memory(received[0], x))
        self.assertFalse(np.shares_memory(received[1], y))


if __name__ == "__main__":
    unittest.main()
