"""JAX-native complete elliptic integrals K(m), E(m) via Carlson symmetric forms.

This module exists because ``jax.scipy.special.ellipk`` / ``ellipe`` are not
exposed by ``jaxlib`` 0.10.0. The blocker
``.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md`` documents the
gap; this helper unblocks the JAX port of ``CircularCoil`` (P1 item 12-sub /
item 15) without taking a host callback.

Implementation
--------------

Carlson symmetric forms ``R_F(x, y, z)`` and ``R_D(x, y, z)`` are evaluated by
the duplication / iterative reduction algorithm of Numerical Recipes 3rd ed.
section 6.11 (page 318, formulas 6.11.4-6.11.6). The Legendre integrals are
written as:

* ``K(m) = R_F(0, 1 - m, 1)``
* ``E(m) = R_F(0, 1 - m, 1) - (m / 3) * R_D(0, 1 - m, 1)``

The duplication recurrence

    lambda    = sqrt(x) sqrt(y) + sqrt(y) sqrt(z) + sqrt(z) sqrt(x)
    (x, y, z) -> ((x + lambda) / 4, (y + lambda) / 4, (z + lambda) / 4)

contracts the arguments toward a common mean. After a fixed number of
duplications a short Taylor series in the centred variables
``d_i = (mu - x_i) / mu`` finishes the evaluation. ``R_D`` carries an
extra running sum ``sigma`` and weight ``fac`` because the integrand is
asymmetric in ``z``.

Iteration count
---------------

``_N_ITER = 12`` doubles the contraction enough that the residual variables
``d_i`` shrink below ~1e-3 even at ``m = 1 - 1e-12``; the truncated NR series
then adds another ~12 decimal digits. An empirical sweep over the full
validation grid (30 evenly spaced ``m`` plus 20 log-spaced near 0 plus 20
log-spaced near 1) finds peak relative error 4.5e-16 for ``K`` and 7.0e-15
for ``E`` versus ``scipy.special``. Both are far inside the
``direct_kernel`` parity lane (``rtol=1e-10, atol=1e-12``).

The loop is expressed with ``jax.lax.scan`` over ``length=_N_ITER`` so the
helper is fully traceable under ``jit``, ``vmap``, and grad transforms with
no host fallback.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Numerical Recipes 6.11: 12 duplications drive the residual variables
# d_i = (mu - x_i)/mu below ~1e-3 even at m = 1 - 1e-12, and the truncated
# Taylor series adds another ~12 decimal digits. Verified empirically across
# the full direct_kernel validation grid; see module docstring.
_N_ITER: int = 12


def _rf_duplication_step(state, _):
    x, y, z = state
    sx = jnp.sqrt(x)
    sy = jnp.sqrt(y)
    sz = jnp.sqrt(z)
    lam = sx * sy + sy * sz + sz * sx
    return (0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)), None


def _carlson_rf(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Carlson symmetric form R_F(x, y, z) (Numerical Recipes 6.11)."""
    (x, y, z), _ = jax.lax.scan(_rf_duplication_step, (x, y, z), None, length=_N_ITER)
    mu = (x + y + z) / 3.0
    dx = (mu - x) / mu
    dy = (mu - y) / mu
    dz = (mu - z) / mu
    # NR 6.11.5 series: 1 - E2/10 + E3/14 + E2^2/24 - 3 E2 E3/44
    #                   - 5 E2^3/208 + 3 E3^2/104 + E2^2 E3/16
    # where E2 = sum_{i<j} d_i d_j, E3 = d_x d_y d_z.
    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz
    series = (
        1.0
        - e2 / 10.0
        + e3 / 14.0
        + e2 * e2 / 24.0
        - 3.0 * e2 * e3 / 44.0
        - 5.0 * e2 * e2 * e2 / 208.0
        + 3.0 * e3 * e3 / 104.0
        + e2 * e2 * e3 / 16.0
    )
    return series / jnp.sqrt(mu)


def _rd_duplication_step(state, _):
    x, y, z, sigma, fac = state
    sx = jnp.sqrt(x)
    sy = jnp.sqrt(y)
    sz = jnp.sqrt(z)
    lam = sx * sy + sy * sz + sz * sx
    sigma_next = sigma + fac / (sz * (z + lam))
    fac_next = 0.25 * fac
    return (
        0.25 * (x + lam),
        0.25 * (y + lam),
        0.25 * (z + lam),
        sigma_next,
        fac_next,
    ), None


def _carlson_rd(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Carlson symmetric form R_D(x, y, z) (Numerical Recipes 6.11)."""
    init = (x, y, z, jnp.zeros_like(x), jnp.ones_like(x))
    (x, y, z, sigma, fac), _ = jax.lax.scan(
        _rd_duplication_step, init, None, length=_N_ITER
    )
    mu = (x + y + 3.0 * z) / 5.0
    dx = (mu - x) / mu
    dy = (mu - y) / mu
    dz = (mu - z) / mu
    # NR 6.11.6: ea = dx*dy, eb = dz^2, ec = ea - eb, ed = ea - 6 eb,
    #            ee = ed + 2 ec.
    ea = dx * dy
    eb = dz * dz
    ec = ea - eb
    ed = ea - 6.0 * eb
    ee = ed + ec + ec
    series = ed * (-3.0 / 14.0 + 9.0 / 88.0 * ed - 4.5 / 26.0 * dz * ee) + dz * (
        ec / 6.0 + dz * (-9.0 / 22.0 * eb + dz * ea * 3.0 / 26.0)
    )
    return 3.0 * sigma + fac * (1.0 + series) / (mu * jnp.sqrt(mu))


@jax.jit
def _ellipk_jit(m: jax.Array) -> jax.Array:
    return _carlson_rf(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))


@jax.jit
def _ellipe_jit(m: jax.Array) -> jax.Array:
    one_minus_m = 1.0 - m
    zeros = jnp.zeros_like(m)
    ones = jnp.ones_like(m)
    rf_val = _carlson_rf(zeros, one_minus_m, ones)
    rd_val = _carlson_rd(zeros, one_minus_m, ones)
    return rf_val - (m / 3.0) * rd_val


def ellipk(m: jax.Array) -> jax.Array:
    """Complete elliptic integral of the first kind, K(m).

    Defined for ``m in [0, 1)`` with ``K(m) = R_F(0, 1 - m, 1)``. K diverges
    logarithmically at ``m = 1``; callers should stay away from that singular
    endpoint, just as the CPU ``CircularCoil`` oracle is singular on the coil
    wire. At ``m = 0`` the integral equals ``pi / 2``.
    """
    return _ellipk_jit(jnp.asarray(m))


def ellipe(m: jax.Array) -> jax.Array:
    """Complete elliptic integral of the second kind, E(m).

    Defined for ``m in [0, 1]`` with
    ``E(m) = R_F(0, 1 - m, 1) - (m / 3) R_D(0, 1 - m, 1)``. At ``m = 0`` the
    integral equals ``pi / 2`` and at ``m = 1`` it equals ``1``.
    """
    return _ellipe_jit(jnp.asarray(m))
