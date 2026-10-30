"""Regression tests for ``set_backend`` process-state changes.

1. In a process whose JAX backends are initialized, GPU allocator settings
   cannot apply: selecting the GPU warns instead of raising and leaves the
   caller's allocator variables alone.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import jax
    import jax.extend.backend
    import simsopt_jax  # noqa: F401
    import simsopt_jax.backend.runtime as runtime_module
    from simsopt_jax.backend.runtime import (
        get_backend_mode,
        set_backend,
    )
import os
import warnings


class TestSetBackendLifecycle(JaxTestCase):
    def test_gpu_selection_after_initialization_keeps_the_allocator_env(self):
        """Selecting the GPU in an initialized process warns about allocator settings
        it cannot apply and leaves the caller's XLA variables unchanged."""
        os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
        os.environ["XLA_FLAGS"] = "--xla_gpu_exclude_nondeterministic_ops=true"
        self.assertTrue(runtime_module._jax_backends_initialized())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_backend("jax", device="gpu")

        self.assertEqual(get_backend_mode(), "jax_gpu_fast")
        self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", os.environ)
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.3")
        messages = [str(warning.message) for warning in caught]
        self.assertTrue(
            any("XLA_PYTHON_CLIENT_PREALLOCATE" in text for text in messages),
            messages,
        )
        if {"cuda", "gpu"} & set(jax.extend.backend.backends()):
            self.assertEqual(jax.numpy.zeros(3).devices(), {jax.devices("gpu")[0]})
