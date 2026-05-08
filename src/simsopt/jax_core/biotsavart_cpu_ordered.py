"""CPU-ordered JAX twins for the Biot-Savart kernel.

The production fast path in :mod:`simsopt.jax_core.biotsavart` uses
``einsum`` / pairwise reductions and computes ``r_inv3`` as
``r_inv * inv(r2)``. The C++ oracle in ``src/simsoptpp/biot_savart_impl.h``
uses a sequential ``lax.fori_loop``-equivalent over quadrature points and
``r_inv * r_inv * r_inv``. Per
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` Phase 3,
this module supplies parity twins that mirror the C++ algebra
operator-for-operator:

* ``diff = point - gamma`` with the C++ sign convention;
* ``norm_diff_3_inv = r_inv * r_inv * r_inv``;
* ``cross = dgamma × diff`` with C++ operand order;
* serial ``lax.fori_loop`` over quadrature points (no XLA pairwise tree);
* ``fak = 1e-7 / num_quad_points`` applied once at output;
* ``B = sum_c (currents[c] * B_per_coil[c])`` accumulated sequentially.

Pure JAX, no ``simsoptpp`` import (M1 contract). Only the parity backend
routes through these kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ._math_utils import (
    explicit_rsqrt as _explicit_rsqrt,
)


__all__ = (
    "biot_savart_B_cpu_ordered",
    "biot_savart_B_and_dB_cpu_ordered",
)


_BIOT_SAVART_FAKMU0 = 1.0e-7


def _per_point_B_one_coil_cpu_ordered(point, gammas, gammadashs):
    """Match ``biot_savart_kernel<derivs=0>`` for a single coil at one point.

    Accumulates B sequentially over quadrature points using the C++ algebra:
    ``r_inv = rsqrt(|diff|²); r_inv³ = r_inv·r_inv·r_inv``,
    ``cross = dgamma × diff``, ``B += cross·r_inv³``. Returns the un-scaled
    sum (the ``fak = 1e-7 / nq`` factor is applied once at the outer
    aggregation).
    """
    nq = gammas.shape[0]
    zero = jnp.zeros((3,), dtype=gammas.dtype)

    def body(j, acc):
        diff = point - gammas[j]
        norm_diff_2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
        norm_diff_inv = _explicit_rsqrt(norm_diff_2)
        norm_diff_3_inv = norm_diff_inv * norm_diff_inv * norm_diff_inv
        # cross(dgamma, diff) — C++ operand order:
        #   c.x = dgamma.y*diff.z - dgamma.z*diff.y
        #   c.y = dgamma.z*diff.x - dgamma.x*diff.z
        #   c.z = dgamma.x*diff.y - dgamma.y*diff.x
        dg = gammadashs[j]
        cx = dg[1] * diff[2] - dg[2] * diff[1]
        cy = dg[2] * diff[0] - dg[0] * diff[2]
        cz = dg[0] * diff[1] - dg[1] * diff[0]
        cross = jnp.array([cx, cy, cz], dtype=gammas.dtype)
        return acc + cross * norm_diff_3_inv

    return lax.fori_loop(0, nq, body, zero)


def _per_point_B_and_dB_one_coil_cpu_ordered(point, gammas, gammadashs):
    """Match ``biot_savart_kernel<derivs=1>`` for a single coil at one point.

    Returns ``(B_acc, dB_acc)`` where the ``fak`` factor is applied later.
    """
    nq = gammas.shape[0]
    zero3 = jnp.zeros((3,), dtype=gammas.dtype)
    zero33 = jnp.zeros((3, 3), dtype=gammas.dtype)

    def body(j, carry):
        B_acc, dB_acc = carry
        diff = point - gammas[j]
        norm_diff_2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
        norm_diff_inv = _explicit_rsqrt(norm_diff_2)
        norm_diff_3_inv = norm_diff_inv * norm_diff_inv * norm_diff_inv
        norm_diff_4_inv = norm_diff_3_inv * norm_diff_inv
        norm_diff = norm_diff_2 * norm_diff_inv  # = ||diff||
        dg = gammadashs[j]
        cx = dg[1] * diff[2] - dg[2] * diff[1]
        cy = dg[2] * diff[0] - dg[0] * diff[2]
        cz = dg[0] * diff[1] - dg[1] * diff[0]
        cross = jnp.array([cx, cy, cz], dtype=gammas.dtype)
        B_acc = B_acc + cross * norm_diff_3_inv

        # Derivative: per C++ biot_savart_impl.h:75–97
        # three_dgamma_by_dphi_cross_diff_by_norm_diff = cross * (3 * norm_diff_inv)
        # numerator1 = cross(dgamma_norm_diff, e_k)
        # tempk = numerator1 - three_cross_inv * diff[k]
        # dB_dX[k] += tempk * norm_diff_4_inv
        three_inv = 3.0 * norm_diff_inv
        three_cross_x = cx * three_inv
        three_cross_y = cy * three_inv
        three_cross_z = cz * three_inv
        # dgamma * norm_diff
        dgnd_x = dg[0] * norm_diff
        dgnd_y = dg[1] * norm_diff
        dgnd_z = dg[2] * norm_diff
        # cross(dgamma_norm_diff, e_k) for k=0,1,2 — equals (0, dgnd.z, -dgnd.y),
        # (-dgnd.z, 0, dgnd.x), (dgnd.y, -dgnd.x, 0).
        # cross(v, e_0) = (v.y*0 - v.z*0, v.z*1 - v.x*0, v.x*0 - v.y*1) = (0, v.z, -v.y)
        # cross(v, e_1) = (v.y*0 - v.z*1, v.z*0 - v.x*0, v.x*1 - v.y*0) = (-v.z, 0, v.x)
        # cross(v, e_2) = (v.y*1 - v.z*0, v.z*0 - v.x*1, v.x*0 - v.y*0) = (v.y, -v.x, 0)
        n0 = jnp.array([0.0, dgnd_z, -dgnd_y], dtype=gammas.dtype)
        n1 = jnp.array([-dgnd_z, 0.0, dgnd_x], dtype=gammas.dtype)
        n2 = jnp.array([dgnd_y, -dgnd_x, 0.0], dtype=gammas.dtype)

        diff0, diff1, diff2 = diff[0], diff[1], diff[2]
        three_cross = jnp.array(
            [three_cross_x, three_cross_y, three_cross_z], dtype=gammas.dtype
        )
        temp0 = n0 - three_cross * diff0
        temp1 = n1 - three_cross * diff1
        temp2 = n2 - three_cross * diff2

        contribution = jnp.stack([temp0, temp1, temp2]) * norm_diff_4_inv
        dB_acc = dB_acc + contribution
        return B_acc, dB_acc

    return lax.fori_loop(0, nq, body, (zero3, zero33))


def _per_point_B_cpu_ordered_summed(point, gammas, gammadashs, currents):
    """Sequential coil sum for B at a single observation point.

    Matches the C++ ``B = sum_i currents[i] * B_per_coil[i]`` in
    ``magneticfield_biotsavart.cpp:62``. ``fak = 1e-7 / nq`` is applied
    once at the end.
    """
    gammas = jnp.asarray(gammas)
    gammadashs = jnp.asarray(gammadashs)
    currents = jnp.asarray(currents)
    ncoils = gammas.shape[0]
    nq = gammas.shape[1]

    def coil_body(c, acc):
        B_one = _per_point_B_one_coil_cpu_ordered(point, gammas[c], gammadashs[c])
        return acc + currents[c] * B_one

    zero = jnp.zeros((3,), dtype=gammas.dtype)
    summed = lax.fori_loop(0, ncoils, coil_body, zero)
    fak = jnp.asarray(_BIOT_SAVART_FAKMU0 / nq, dtype=gammas.dtype)
    return fak * summed


def _per_point_B_and_dB_cpu_ordered_summed(point, gammas, gammadashs, currents):
    gammas = jnp.asarray(gammas)
    gammadashs = jnp.asarray(gammadashs)
    currents = jnp.asarray(currents)
    ncoils = gammas.shape[0]
    nq = gammas.shape[1]
    zero3 = jnp.zeros((3,), dtype=gammas.dtype)
    zero33 = jnp.zeros((3, 3), dtype=gammas.dtype)

    def coil_body(c, carry):
        B_acc, dB_acc = carry
        B_one, dB_one = _per_point_B_and_dB_one_coil_cpu_ordered(
            point, gammas[c], gammadashs[c]
        )
        return B_acc + currents[c] * B_one, dB_acc + currents[c] * dB_one

    B_summed, dB_summed = lax.fori_loop(0, ncoils, coil_body, (zero3, zero33))
    fak = jnp.asarray(_BIOT_SAVART_FAKMU0 / nq, dtype=gammas.dtype)
    return fak * B_summed, fak * dB_summed


def biot_savart_B_cpu_ordered(points, gammas, gammadashs, currents):
    """Sequential, C++-order Biot-Savart B kernel for one coil group.

    Args:
        points: (npts, 3) observation points.
        gammas: (ncoils, nq, 3) coil geometry.
        gammadashs: (ncoils, nq, 3) ``dgamma_by_dphi``.
        currents: (ncoils,) coil currents.

    Returns:
        B: (npts, 3).
    """
    return jax.vmap(_per_point_B_cpu_ordered_summed, in_axes=(0, None, None, None))(
        points, gammas, gammadashs, currents
    )


def biot_savart_B_and_dB_cpu_ordered(points, gammas, gammadashs, currents):
    """Sequential, C++-order Biot-Savart kernel for B and dB/dX (one coil group).

    Returns:
        (B, dB_dX) with shapes (npts, 3) and (npts, 3, 3).
    """
    return jax.vmap(
        _per_point_B_and_dB_cpu_ordered_summed,
        in_axes=(0, None, None, None),
    )(points, gammas, gammadashs, currents)
