import unittest

import jax.numpy as jnp
import numpy as np
from jax import lax

from simsopt.geo.jit import jit


def _aligned(n):
    """A 64-byte aligned array, which the CPU backend may use without copying."""
    storage = np.empty(n + 8)
    offset = (-storage.ctypes.data % 64) // storage.itemsize
    array = storage[offset:offset + n]
    array[:] = 1.0
    return array


@jit
def _sum_after_work(x, matrix):
    product = lax.fori_loop(0, 4, lambda _, y: (y @ matrix) / matrix.shape[0], matrix)
    return jnp.sum(jnp.where(product[0, 0] > 0, x, -x))


class JitTests(unittest.TestCase):
    def test_numpy_argument_is_copied_before_dispatch(self):
        """Changing a NumPy argument after the call cannot change the result.

        Native curves return views of buffers that the next DOF change overwrites.
        """
        x = _aligned(1024)
        matrix = jnp.ones((2048, 2048), dtype=jnp.float32)
        _sum_after_work(x, matrix).block_until_ready()
        result = _sum_after_work(x, matrix)
        self.assertFalse(result.is_ready(), "evaluation finished before the argument changed")
        x[:] = -7.0
        self.assertEqual(float(result), 1024.0)


if __name__ == "__main__":
    unittest.main()
