"""Item 12 Carlson elliptic-helper parity tests.

``simsopt.jax_core._elliptic`` provides JAX-native replacements for the
SciPy complete elliptic integrals used by ``CircularCoil``. These tests
compare the helper surface directly against ``scipy.special`` over the
parameter convention ``m = k**2`` used by both SciPy and the upstream
``CircularCoil`` CPU oracle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core._elliptic import ellipe, ellipk


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _elliptic_m_values() -> np.ndarray:
    eps = np.finfo(np.float64).eps
    return np.asarray(
        [
            0.0,
            eps,
            1.0e-16,
            1.0e-12,
            1.0e-9,
            1.0e-6,
            1.0e-3,
            0.1,
            0.5,
            0.9,
            0.999,
            1.0 - 1.0e-9,
            1.0 - 1.0e-12,
            1.0 - eps,
        ],
        dtype=np.float64,
    )


def test_ellipk_matches_scipy_over_stress_points():
    m = _elliptic_m_values()
    np.testing.assert_allclose(
        np.asarray(ellipk(m), dtype=np.float64),
        special.ellipk(m),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_ellipe_matches_scipy_over_stress_points():
    m = _elliptic_m_values()
    np.testing.assert_allclose(
        np.asarray(ellipe(m), dtype=np.float64),
        special.ellipe(m),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_elliptic_helpers_run_under_strict_transfer_guard():
    m_dev = jnp.asarray(_elliptic_m_values(), dtype=jnp.float64)
    m_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        ellipk(m_dev).block_until_ready()
        ellipe(m_dev).block_until_ready()
