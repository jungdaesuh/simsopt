"""Compatibility shim for pure JAX surface Fourier kernels.

The implementation lives in :mod:`simsopt_jax.core.surface_fourier_kernels`.
This module preserves the historical public import path.
"""

from simsopt_jax.core.surface_fourier_kernels import *  # noqa: F403
from simsopt_jax.core.surface_fourier_kernels import __all__ as __all__
from simsopt_jax.core.surface_fourier_kernels import (
    _dofs_to_xyzc_any as _dofs_to_xyzc_any,
    _two_pi as _two_pi,
    _unitnormal as _unitnormal,
)
