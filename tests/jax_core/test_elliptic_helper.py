"""Parity tests for ``simsopt.jax_core._elliptic`` against ``scipy.special``.

The Carlson R_F / R_D helpers implement complete elliptic integrals K(m) and
E(m) without a host fallback because ``jax.scipy.special.ellipk`` / ``ellipe``
are not exposed by ``jaxlib`` 0.10.0; the gap is recorded in
``.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md``.

Parity is gated at the ``direct_kernel`` lane (``rtol=1e-10, atol=1e-12``)
across three regimes: evenly spaced ``m in [0, 1 - 1e-12]``, log-spaced
``m`` near 0, and log-spaced ``m`` near 1. The transfer-guard test confirms
the JAX implementation runs cleanly under
``jax.transfer_guard("disallow")`` so the helper can be embedded inside the
CircularCoil JAX kernel without producing implicit host transfers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core._elliptic import ellipe, ellipk

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _grid() -> np.ndarray:
    even = np.linspace(0.0, 1.0 - 1e-12, 30)
    near_zero = np.logspace(-12, -2, 20)
    near_one = 1.0 - np.logspace(-12, -3, 20)
    return np.concatenate([even, near_zero, near_one]).astype(np.float64)


# -- scalar parity ---------------------------------------------------------


def test_ellipk_parity_vs_scipy_full_grid():
    """K(m) parity at the ``direct_kernel`` lane over the full validation grid.

    Oracle: ``scipy.special.ellipk``. The Carlson R_F implementation is
    expected to match within machine epsilon, far inside the ``direct_kernel``
    tolerances.
    """
    grid = _grid()
    ref = scipy.special.ellipk(grid)
    got = np.asarray(jax.vmap(ellipk)(jnp.asarray(grid)), dtype=np.float64)
    np.testing.assert_allclose(got, ref, rtol=_RTOL, atol=_ATOL)


def test_ellipe_parity_vs_scipy_full_grid():
    """E(m) parity at the ``direct_kernel`` lane over the full validation grid.

    Oracle: ``scipy.special.ellipe``. Composition ``R_F - (m/3) R_D`` is
    expected to match within machine epsilon.
    """
    grid = _grid()
    ref = scipy.special.ellipe(grid)
    got = np.asarray(jax.vmap(ellipe)(jnp.asarray(grid)), dtype=np.float64)
    np.testing.assert_allclose(got, ref, rtol=_RTOL, atol=_ATOL)


def test_ellipk_zero_endpoint_exact():
    """``K(0) = pi/2`` within machine epsilon (Carlson series limit)."""
    got = float(ellipk(jnp.array(0.0)))
    assert abs(got - np.pi / 2) < _ATOL


def test_ellipe_zero_endpoint_exact():
    """``E(0) = pi/2`` within machine epsilon (Carlson series limit)."""
    got = float(ellipe(jnp.array(0.0)))
    assert abs(got - np.pi / 2) < _ATOL


# -- vmap consistency ------------------------------------------------------


def test_ellipk_vmap_matches_scalar():
    """Vmapped K(m) matches the scalar API element-wise.

    Confirms the helper composes with ``jax.vmap`` over leading batch
    dimensions without internal Python loops.
    """
    grid = _grid()
    m_arr = jnp.asarray(grid)
    vmapped = np.asarray(jax.vmap(ellipk)(m_arr), dtype=np.float64)
    scalar = np.asarray(
        [float(ellipk(jnp.asarray(value))) for value in grid], dtype=np.float64
    )
    np.testing.assert_array_equal(vmapped, scalar)


def test_ellipe_vmap_matches_scalar():
    """Vmapped E(m) matches the scalar API element-wise."""
    grid = _grid()
    m_arr = jnp.asarray(grid)
    vmapped = np.asarray(jax.vmap(ellipe)(m_arr), dtype=np.float64)
    scalar = np.asarray(
        [float(ellipe(jnp.asarray(value))) for value in grid], dtype=np.float64
    )
    np.testing.assert_array_equal(vmapped, scalar)


# -- JIT consistency -------------------------------------------------------


def test_ellipk_jit_matches_eager():
    """``jit(ellipk)(m)`` returns bit-identical values to the eager call."""
    grid = _grid()
    m_arr = jnp.asarray(grid)
    eager = jax.vmap(ellipk)(m_arr)
    jitted = jax.jit(jax.vmap(ellipk))(m_arr)
    np.testing.assert_array_equal(
        np.asarray(eager, dtype=np.float64), np.asarray(jitted, dtype=np.float64)
    )


def test_ellipe_jit_matches_eager():
    """``jit(ellipe)(m)`` returns bit-identical values to the eager call."""
    grid = _grid()
    m_arr = jnp.asarray(grid)
    eager = jax.vmap(ellipe)(m_arr)
    jitted = jax.jit(jax.vmap(ellipe))(m_arr)
    np.testing.assert_array_equal(
        np.asarray(eager, dtype=np.float64), np.asarray(jitted, dtype=np.float64)
    )


# -- Transfer-guard discipline --------------------------------------------


def test_ellipk_ellipe_under_strict_transfer_guard():
    """Both helpers run under ``transfer_guard('disallow')`` without host hops.

    The CircularCoil JAX kernel will call ``ellipk`` and ``ellipe`` inside
    JIT-compiled paths under ``jax.transfer_guard("disallow")``; this test
    locks that contract in for the helper itself.
    """
    grid = _grid()
    m_dev = jnp.asarray(grid, dtype=jnp.float64)
    m_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        jax.vmap(ellipk)(m_dev).block_until_ready()
        jax.vmap(ellipe)(m_dev).block_until_ready()
        jax.jit(jax.vmap(ellipk))(m_dev).block_until_ready()
        jax.jit(jax.vmap(ellipe))(m_dev).block_until_ready()
