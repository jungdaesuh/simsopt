"""Parity tests for ``simsopt_jax.core._elliptic`` against ``scipy.special``.

The Carlson R_F / R_D helpers implement the complete elliptic integrals K(m)
and E(m) used by ``CircularCoil`` with no host fallback, in the parameter
convention ``m = k**2`` shared by SciPy and the upstream ``CircularCoil`` CPU
oracle. Parity is gated at the ``direct_kernel`` lane over four grids: stress
points down to machine epsilon at both ends, evenly spaced ``m``, and
log-spaced ``m`` near 0 and near 1.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax.core._elliptic import ellipe, ellipk

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]
_EPS = np.finfo(np.float64).eps

_M_GRIDS = {
    "stress_points": np.asarray(
        [
            0.0,
            _EPS,
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
            1.0 - _EPS,
        ],
        dtype=np.float64,
    ),
    "even": np.linspace(0.0, 1.0 - 1e-12, 30, dtype=np.float64),
    "near_zero": np.logspace(-12, -2, 20, dtype=np.float64),
    "near_one": 1.0 - np.logspace(-12, -3, 20, dtype=np.float64),
}
_ALL_M = np.concatenate(tuple(_M_GRIDS.values()))

_INTEGRALS = {
    "ellipk": (ellipk, special.ellipk),
    "ellipe": (ellipe, special.ellipe),
}


@pytest.mark.parametrize("grid", sorted(_M_GRIDS))
@pytest.mark.parametrize("integral", sorted(_INTEGRALS))
def test_elliptic_integral_matches_scipy(integral: str, grid: str) -> None:
    """Batched and vmapped evaluations both match SciPy at ``direct_kernel``."""
    helper, oracle = _INTEGRALS[integral]
    m = _M_GRIDS[grid]
    reference = oracle(m)
    m_dev = jnp.asarray(m)

    for evaluated in (helper(m_dev), jax.vmap(helper)(m_dev)):
        np.testing.assert_allclose(
            np.asarray(evaluated, dtype=np.float64),
            reference,
            rtol=_RTOL,
            atol=_ATOL,
        )


@pytest.mark.parametrize(
    ("integral", "m", "expected"),
    [
        ("ellipk", 0.0, np.pi / 2),
        ("ellipe", 0.0, np.pi / 2),
        ("ellipk", 1.0, np.inf),
        ("ellipe", 1.0, 1.0),
    ],
)
def test_elliptic_integral_endpoints(integral: str, m: float, expected: float) -> None:
    """``K(0) = E(0) = pi/2``, ``E(1) = 1`` and SciPy's singular ``K(1) = inf``."""
    helper, _oracle = _INTEGRALS[integral]
    got = float(helper(jnp.array(m)))
    if np.isinf(expected):
        assert np.isposinf(got)
    else:
        assert abs(got - expected) < _ATOL


@pytest.mark.parametrize("integral", sorted(_INTEGRALS))
def test_elliptic_integral_vmap_and_jit_match_the_scalar_call(integral: str) -> None:
    """``vmap`` and ``jit(vmap)`` are bit-identical to element-wise scalar calls."""
    helper, _oracle = _INTEGRALS[integral]
    m_dev = jnp.asarray(_ALL_M)
    scalar = np.asarray(
        [float(helper(jnp.asarray(value))) for value in _ALL_M], dtype=np.float64
    )

    np.testing.assert_array_equal(
        np.asarray(jax.vmap(helper)(m_dev), dtype=np.float64), scalar
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jit(jax.vmap(helper))(m_dev), dtype=np.float64), scalar
    )


def test_elliptic_integrals_run_under_strict_transfer_guard() -> None:
    """Batched, vmapped and jitted calls make no implicit host transfer.

    The CircularCoil JAX kernel calls both helpers inside compiled paths under
    ``jax.transfer_guard("disallow")``.
    """
    m_dev = jnp.asarray(_ALL_M, dtype=jnp.float64)
    m_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        for helper in (ellipk, ellipe):
            helper(m_dev).block_until_ready()
            jax.vmap(helper)(m_dev).block_until_ready()
            jax.jit(jax.vmap(helper))(m_dev).block_until_ready()
