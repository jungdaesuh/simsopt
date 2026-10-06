"""Run as ``python -m jax_test_support_import_probe`` with ``tests`` on ``PYTHONPATH``.

Importing ``jax_test_support`` must pin XLA's CUDA autotuners in ``XLA_FLAGS``
and force ``jax_enable_x64``: the JAX test runtime, applied by its first import.
The parent starts this process with no ``XLA_FLAGS`` and without JAX imported.
"""

from __future__ import annotations

import os
import sys

assert "jax" not in sys.modules
assert "XLA_FLAGS" not in os.environ

import jax_test_support  # noqa: F401 - the import under test

import jax
from simsopt_jax.backend import runtime

assert os.environ["XLA_FLAGS"] == (
    f"{runtime._GPU_FUSION_AUTOTUNER_DISABLED} {runtime._GPU_AUTOTUNE_LEVEL_PINNED}"
)
assert jax.config.jax_enable_x64 is True
print("jax_test_support import applies the JAX test runtime")
