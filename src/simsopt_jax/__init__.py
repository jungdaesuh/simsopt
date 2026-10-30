"""Canonical namespace for Simsopt JAX implementations.

Requires Python 3.11 or newer, like the JAX 0.10 it uses (``pip install simsopt[jax]``).
"""

# _requirements (the Python and JAX version gate) must load first: tests treat
# an ImportError raised after it has loaded as a broken package, not an
# unsupported runtime.
from . import _requirements, core, runtime

__all__ = ("core", "runtime")
