"""Compatibility shim for pure JAX surface Fourier kernels.

The implementation lives in :mod:`simsopt.jax_core.surface_fourier_kernels`.
This module preserves the historical public import path.
"""

from simsopt.jax_core.surface_fourier_kernels import *  # noqa: F403
from simsopt.jax_core.surface_fourier_kernels import __all__ as __all__
from simsopt.jax_core.surface_fourier_kernels import _unitnormal as _unitnormal
