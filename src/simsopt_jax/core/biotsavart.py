"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
evaluate Biot-Savart on CPU or GPU without a native field object.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import cast

import jax
from simsopt_jax.backend import get_field_kernel_tuning, register_backend_cache_clear
from jax import lax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.runtime.host_boundary import snapshot_host_tree


from ._device_scalars import device_one as _device_one
from ._math_utils import (
    as_compute_array as _as_compute_array,
    as_jax_float64 as _as_jax_float64,
    axis0_entries as _axis0_entries,
    explicit_inv as _explicit_inv,
    explicit_rsqrt as _explicit_rsqrt,
    eye as _eye,
    scalar_like as _scalar_like,
    zeros as _zeros,
)
from .reductions import pairwise_sum_axis as _pairwise_sum_axis

__all__ = [
    "biot_savart_B",
    "biot_savart_B_vjp",
    "biot_savart_dB_by_dX",
    "biot_savart_d2B_by_dXdX",
    "biot_savart_d2B_by_dXdX_vjp",
    "biot_savart_B_and_dB",
    "biot_savart_A",
    "biot_savart_dA_by_dX",
    "biot_savart_d2A_by_dXdX",
    "group_coil_data",
    "invalidate_kernel_cache",
]

_MU0_OVER_4PI = 1e-7


# ── Config reader (monkeypatchable by tests) ──────────────────────────


def _read_tuning_config() -> tuple:
    """Return ``(coil_chunk_size, quadrature_block_size, point_chunk_size)``.

    Forward factories and the B pullback use these three tuning values.
    Tests override this reader (+ ``invalidate_kernel_cache``) rather than
    patching three stubs. The Hessian VJP reads its independent runtime tile.

    Resolves the backend config once to avoid repeated mode/policy lookups.
    """
    fkt = get_field_kernel_tuning()
    return fkt.coil_chunk_size, fkt.quadrature_block_size, fkt.point_chunk_size


# ── Array slicing primitives ──────────────────────────────────────────


def _slice_coil_chunk(array, start, chunk_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        array,
        (_as_int32_scalar(start),) + (zero,) * (array.ndim - 1),
        (chunk_size,) + tuple(array.shape[1:]),
    )


def _slice_quadrature_block(array, start, block_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        array,
        (zero, _as_int32_scalar(start)) + (zero,) * (array.ndim - 2),
        (array.shape[0], block_size) + tuple(array.shape[2:]),
    )


def _slice_point_chunk(points, start, chunk_size: int):
    zero = _as_int32_scalar(0)
    return lax.dynamic_slice(
        points,
        (_as_int32_scalar(start),) + (zero,) * (points.ndim - 1),
        (chunk_size,) + tuple(points.shape[1:]),
    )


def _float64_scalar(reference, value):
    return _device_one(reference) * np.float64(value)


def _as_int32_scalar(value):
    return jnp.asarray(value, dtype=jnp.int32)


def _radius_squared(diff):
    """Return exact ``r^2`` so point-on-coil singularities stay loud."""

    return jnp.sum(diff * diff, axis=-1)


def _cross_product(left, right):
    left_x = left[..., 0]
    left_y = left[..., 1]
    left_z = left[..., 2]
    right_x = right[..., 0]
    right_y = right[..., 1]
    right_z = right[..., 2]
    return jnp.stack(
        (
            left_y * right_z - left_z * right_y,
            left_z * right_x - left_x * right_z,
            left_x * right_y - left_y * right_x,
        ),
        axis=-1,
    )


# ── Tree utilities ────────────────────────────────────────────────────


def _tree_dynamic_update(prefix_tree, chunk_tree, start_index: int):
    return jax.tree.map(
        lambda acc, update: lax.dynamic_update_slice(
            acc,
            update,
            (_as_int32_scalar(start_index),) + (_as_int32_scalar(0),) * (acc.ndim - 1),
        ),
        prefix_tree,
        chunk_tree,
    )


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree.map(
        lambda leaf: _zeros(
            (prefix_size,) + tuple(leaf.shape[1:]),
            dtype=leaf.dtype,
        ),
        reference_tree,
    )


def _tree_add(left, right):
    return jax.tree.map(
        lambda left_leaf, right_leaf: left_leaf + right_leaf, left, right
    )


# ── Tiling primitives ────────────────────────────────────────────────


def _two_chunk_sum(first_chunk, second_chunk, reduce_chunk):
    return _tree_add(reduce_chunk(*first_chunk), reduce_chunk(*second_chunk))


def _coil_chunk_reduce(
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    zero,
    reduce_chunk,
):
    """Reduce coil chunks assuming a concrete leading coil dimension.

    This helper is only safe on call paths where ``currents.shape[0]`` is known
    at trace time. The surrounding kernel builders satisfy that contract by
    specializing each JIT closure to a fixed grouped-coil layout before calling
    into this reducer.
    """
    coil_count = currents.shape[0]
    if coil_count == 0:
        return zero
    if chunk_size <= 0 or coil_count <= chunk_size:
        return reduce_chunk(gammas, gammadashs, currents)

    chunk_count = (coil_count + chunk_size - 1) // chunk_size
    if chunk_count == 2:
        # Exact two-chunk shapes avoid padded loop overhead in B/B+dB hot paths.
        second_chunk_size = coil_count - chunk_size
        return _two_chunk_sum(
            (
                _slice_coil_chunk(gammas, 0, chunk_size),
                _slice_coil_chunk(gammadashs, 0, chunk_size),
                lax.dynamic_slice(currents, (0,), (chunk_size,)),
            ),
            (
                _slice_coil_chunk(gammas, chunk_size, second_chunk_size),
                _slice_coil_chunk(gammadashs, chunk_size, second_chunk_size),
                lax.dynamic_slice(currents, (chunk_size,), (second_chunk_size,)),
            ),
            reduce_chunk,
        )

    full_chunk_count, tail_size = divmod(coil_count, chunk_size)

    # Keep the outer coil accumulation serial until parity data implicates it;
    # the quadrature-axis sum dominates the known reduction-order drift, while
    # tree-reducing chunk outputs would add another staged hot-path combine.
    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_gammas = _slice_coil_chunk(gammas, start, chunk_size)
        chunk_gammadashs = _slice_coil_chunk(gammadashs, start, chunk_size)
        chunk_currents = lax.dynamic_slice(
            currents,
            (start,),
            (chunk_size,),
        )
        return _tree_add(
            acc,
            reduce_chunk(chunk_gammas, chunk_gammadashs, chunk_currents),
        )

    reduced = lax.fori_loop(0, full_chunk_count, body, zero)
    if tail_size:
        tail_start = full_chunk_count * chunk_size
        reduced = _tree_add(
            reduced,
            reduce_chunk(
                gammas[tail_start:],
                gammadashs[tail_start:],
                currents[tail_start:],
            ),
        )
    return reduced


def _quadrature_block_integral(
    x,
    gammas,
    gammadashs,
    *,
    block_size: int,
    integrand,
):
    quadrature_count = gammas.shape[1]
    # The single-block, exact two-block, and full-block-plus-tail >=3 paths use
    # different quadrature reduction-tree shapes. Keep parity tests pinned
    # across block boundaries when changing this routine.
    if block_size <= 0 or quadrature_count <= block_size:
        values = integrand(x, gammas, gammadashs)
        return _pairwise_sum_axis(values, axis=1) * _scalar_like(
            values,
            1.0 / values.shape[1],
        )

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
        # Exact two-block shapes avoid padded loop overhead in B/B+dB hot paths.
        second_block_size = quadrature_count - block_size
        return (
            _two_chunk_sum(
                (
                    _slice_quadrature_block(gammas, 0, block_size),
                    _slice_quadrature_block(gammadashs, 0, block_size),
                ),
                (
                    _slice_quadrature_block(gammas, block_size, second_block_size),
                    _slice_quadrature_block(
                        gammadashs,
                        block_size,
                        second_block_size,
                    ),
                ),
                lambda gg, ggd: _pairwise_sum_axis(integrand(x, gg, ggd), axis=1),
            )
            / quadrature_count
        )

    full_block_count, tail_size = divmod(quadrature_count, block_size)
    zero = _zeros((gammas.shape[0], 3), dtype=jnp.float64)

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            gammadashs,
            start,
            block_size,
        )
        block_integrand = integrand(x, block_gammas, block_gammadashs)
        return acc + _pairwise_sum_axis(block_integrand, axis=1)

    integral_sum = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail_start = full_block_count * block_size
        tail_integrand = integrand(
            x,
            gammas[:, tail_start:],
            gammadashs[:, tail_start:],
        )
        integral_sum = integral_sum + _pairwise_sum_axis(tail_integrand, axis=1)
    return integral_sum / quadrature_count


# The Hessian integrand carries 27 doubles per (point, quadrature node) where
# ``B`` and ``dB`` together carry 12, and its ``jacfwd(jacfwd)`` trace keeps
# both tangent levels live.  The tuning table sizes ``point_chunk_size`` for
# the ``B``/``B+dB`` hot paths, so the Hessian kernel takes one eighth of it:
# 2048 points at 18 NCSX coils x 250 nodes otherwise materializes >20 GB and
# fails on a 32 GB device, while 256 points stay near 2 GB.
_HESSIAN_POINT_CHUNK_DIVISOR = 8


def hessian_point_chunk_size(point_chunk_size: int) -> int:
    """Point tile for ``d2B/dXdX`` derived from the ``B``-sized tuning value.

    ``0`` (tiling disabled) is preserved so audit policies that pin every
    chunk to the dense layout keep their contract; any positive size is
    divided by ``_HESSIAN_POINT_CHUNK_DIVISOR`` and floored at one point.

    Args:
        point_chunk_size (int): Point tile size for field values; nonpositive sizes
            disable tiling.

    Returns:
        int: Zero when tiling is disabled; otherwise the value tile divided by
            the Hessian divisor and floored at one.
    """
    if point_chunk_size <= 0:
        return 0
    return max(1, point_chunk_size // _HESSIAN_POINT_CHUNK_DIVISOR)


def _point_chunk_reduce(points, chunk_kernel, chunk_size):
    """Evaluate *chunk_kernel* over *points* with optional point-axis tiling.

    Three-arg signature: ``chunk_size`` is passed explicitly by the kernel
    factory (not read from module state) so that the value is part of the
    ``lru_cache`` key and triggers recompilation when it changes.
    """
    point_count = points.shape[0]
    if point_count == 0 or chunk_size <= 0 or point_count <= chunk_size:
        return chunk_kernel(points)

    full_chunk_count, tail_size = divmod(point_count, chunk_size)
    # Keep the default remat policy until CUDA profiling shows that saving dot
    # residuals is a better memory/runtime tradeoff for this kernel.
    remat_chunk_kernel = jax.checkpoint(chunk_kernel)
    first_chunk_points = _slice_point_chunk(points, 0, chunk_size)
    first_result = remat_chunk_kernel(first_chunk_points)
    result = _tree_dynamic_update(
        _tree_zeros_like_prefix(first_result, point_count),
        first_result,
        0,
    )

    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_points = _slice_point_chunk(points, start, chunk_size)
        chunk_result = remat_chunk_kernel(chunk_points)
        return _tree_dynamic_update(acc, chunk_result, start)

    result = lax.fori_loop(1, full_chunk_count, body, result)
    if tail_size:
        tail_start = full_chunk_count * chunk_size
        tail_result = remat_chunk_kernel(points[tail_start:])
        result = _tree_dynamic_update(result, tail_result, tail_start)
    return result


# ── Physics integrands ────────────────────────────────────────────────


def _biot_savart_B_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = _radius_squared(diff)
    r_inv = _explicit_rsqrt(r2)
    r_inv3 = r_inv * _explicit_inv(r2)
    cross = _cross_product(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = _radius_squared(diff)
    r_inv = _explicit_rsqrt(r2)
    return gammadashs * r_inv[..., None]


def _biot_savart_B_and_dB_integrand(x, gammas, gammadashs):
    diff = x - gammas
    r2 = _radius_squared(diff)
    r_inv = _explicit_rsqrt(r2)
    r_inv3 = r_inv * _explicit_inv(r2)
    r_inv4 = r_inv3 * r_inv
    norm = r2 * r_inv

    cross = _cross_product(gammadashs, diff)
    B_integrand = cross * r_inv3[..., None]

    dg_norm = gammadashs * norm[..., None]
    zero = _zeros(diff.shape[:-1], dtype=diff.dtype)
    numerator0 = jnp.stack((zero, dg_norm[..., 2], -dg_norm[..., 1]), axis=-1)
    numerator1 = jnp.stack((-dg_norm[..., 2], zero, dg_norm[..., 0]), axis=-1)
    numerator2 = jnp.stack((dg_norm[..., 1], -dg_norm[..., 0], zero), axis=-1)

    three_cross_inv = cross * (3.0 * r_inv)[..., None]
    temp0 = numerator0 - three_cross_inv * diff[..., 0, None]
    temp1 = numerator1 - three_cross_inv * diff[..., 1, None]
    temp2 = numerator2 - three_cross_inv * diff[..., 2, None]
    dB_integrand = jnp.stack((temp0, temp1, temp2), axis=-2) * r_inv4[..., None, None]
    return B_integrand, dB_integrand


# The gradgradB pullback contracts a Hessian seed ``W[p, i, j, l]`` with
# ``d_i d_j B_l`` at each point.  Per pair (point p, node k) the integrand is
# ``f(r, t) = t x r / |r|^3`` with ``r = x_p - gamma_k`` and ``t = gamma'_k``,
# and ``d_i d_j f_l = -eps_lmn t_m h_ijn`` with ``h = 1/|r|``.  The contracted
# scalar is therefore ``phi = t . c(r)``, writing ``rho = |r|``, with
#   c(r) = 15 rho^-7 (r x q) - 3 rho^-5 (r x tau + X^T r),
#   q_l = r^T S_l r,  tau_l = W_iil,  X_jm = eps_mnl (W_njl + W_jnl),
# where ``S_l`` is the symmetric part of ``W[:, :, l]``.  Its spatial gradient
#   grad phi = 15 rho^-7 (q x t + 2 sum_l (t x r)_l S_l r) - 105 rho^-9 a r
#              - 3 rho^-5 (tau x t + X t) + 15 rho^-7 (b + d) r,
#   a = (t x r) . q,  b = t . (r x tau),  d = t . X^T r,
# pulls back to ``gamma`` with a minus sign; ``c`` is the ``gamma'`` cotangent
# and ``t . c`` the per-unit-current one.  The closed form avoids reverse mode
# through ``jacfwd(jacfwd(B))`` and the tangent planes it stores.


def _cross3(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot3(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _hessian_seed_point_terms(vgradgrad):
    """Per-point ``(S, tau, X)`` from a ``[point, d1, d2, component]`` seed."""
    symmetric_sum = vgradgrad + jnp.swapaxes(vgradgrad, 1, 2)
    trace = jnp.trace(vgradgrad, axis1=1, axis2=2)
    axial = jnp.stack(
        (
            symmetric_sum[:, 1, :, 2] - symmetric_sum[:, 2, :, 1],
            symmetric_sum[:, 2, :, 0] - symmetric_sum[:, 0, :, 2],
            symmetric_sum[:, 0, :, 1] - symmetric_sum[:, 1, :, 0],
        ),
        axis=-1,
    )
    return 0.5 * symmetric_sum, trace, axial


def _hessian_vjp_tile_sums(points, vgradgrad, gammas, gammadashs):
    """Point sums of ``grad phi`` and ``c`` per node, each of shape ``(C, Q, 3)``.

    Unscaled by currents, ``mu0/4pi`` and the quadrature weight; see the
    closed form above.
    """
    seed_sym, seed_trace, seed_axial = _hessian_seed_point_terms(vgradgrad)

    def per_point(values):
        return values[:, None, None]

    r = tuple(per_point(points[:, i]) - gammas[None, :, :, i] for i in range(3))
    t = tuple(gammadashs[None, :, :, i] for i in range(3))
    tau = tuple(per_point(seed_trace[:, l]) for l in range(3))
    r2 = _dot3(r, r)
    r_inv2 = _explicit_inv(r2)
    r_inv5 = _explicit_rsqrt(r2) * r_inv2 * r_inv2
    r_inv7 = r_inv5 * r_inv2
    r_inv9 = r_inv7 * r_inv2
    sym_r = tuple(
        tuple(
            sum(per_point(seed_sym[:, i, j, l]) * r[j] for j in range(3))
            for i in range(3)
        )
        for l in range(3)
    )
    q = tuple(_dot3(r, sym_r[l]) for l in range(3))
    axial_r = tuple(
        sum(per_point(seed_axial[:, j, m]) * r[j] for j in range(3)) for m in range(3)
    )
    axial_t = tuple(
        sum(per_point(seed_axial[:, j, m]) * t[m] for m in range(3)) for j in range(3)
    )
    t_cross_r = _cross3(t, r)
    a = _dot3(t_cross_r, q)
    b_plus_d = _dot3(t, _cross3(r, tau)) + _dot3(t, axial_r)
    r_cross_q = _cross3(r, q)
    r_cross_tau = _cross3(r, tau)
    q_cross_t = _cross3(q, t)
    tau_cross_t = _cross3(tau, t)
    c = tuple(
        15.0 * r_inv7 * r_cross_q[m] - 3.0 * r_inv5 * (r_cross_tau[m] + axial_r[m])
        for m in range(3)
    )
    grad = tuple(
        15.0 * r_inv7 * (q_cross_t[k] + 2.0 * _dot3(t_cross_r, tuple(sym_r[l][k] for l in range(3))))
        + (15.0 * r_inv7 * b_plus_d - 105.0 * r_inv9 * a) * r[k]
        - 3.0 * r_inv5 * (tau_cross_t[k] + axial_t[k])
        for k in range(3)
    )
    return (
        jnp.stack([jnp.sum(component, axis=0) for component in grad], axis=-1),
        jnp.stack([jnp.sum(component, axis=0) for component in c], axis=-1),
    )


# ── Dense one-point kernels ──────────────────────────────────────────


class _Integrand(Enum):
    B = "B"
    A = "A"


class _DiffMode(Enum):
    VALUE = "value"
    JACOBIAN = "jacobian"
    HESSIAN = "hessian"
    VALUE_AND_JACOBIAN = "value_and_jacobian"


_INTEGRANDS = {
    _Integrand.B: _biot_savart_B_integrand,
    _Integrand.A: _biot_savart_A_integrand,
}


def _one_point_dense(
    x, gammas, gammadashs, currents, *, integrand, quadrature_block_size=0
):
    integral = _quadrature_block_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
        integrand=integrand,
    )
    return _float64_scalar(currents, _MU0_OVER_4PI) * jnp.einsum(
        "c,cj->j", currents, integral, precision=lax.Precision.HIGHEST
    )


def _quadrature_block_B_and_dB_integral(x, gammas, gammadashs, *, block_size: int):
    quadrature_count = gammas.shape[1]

    def reduce_values(block_gammas, block_gammadashs):
        return jax.tree.map(
            lambda values: _pairwise_sum_axis(values, axis=1),
            _biot_savart_B_and_dB_integrand(x, block_gammas, block_gammadashs),
        )

    def average_sum(integral_sum):
        return jax.tree.map(lambda values: values / quadrature_count, integral_sum)

    if block_size <= 0 or quadrature_count <= block_size:
        return average_sum(reduce_values(gammas, gammadashs))

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
        second_block_size = quadrature_count - block_size
        first = reduce_values(
            _slice_quadrature_block(gammas, 0, block_size),
            _slice_quadrature_block(gammadashs, 0, block_size),
        )
        second = reduce_values(
            _slice_quadrature_block(gammas, block_size, second_block_size),
            _slice_quadrature_block(gammadashs, block_size, second_block_size),
        )
        return average_sum(_tree_add(first, second))

    full_block_count, tail_size = divmod(quadrature_count, block_size)
    zero = (
        _zeros((gammas.shape[0], 3), dtype=jnp.float64),
        _zeros((gammas.shape[0], 3, 3), dtype=jnp.float64),
    )

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            gammadashs,
            start,
            block_size,
        )
        return _tree_add(acc, reduce_values(block_gammas, block_gammadashs))

    integral_sum = lax.fori_loop(0, full_block_count, body, zero)
    if tail_size:
        tail_start = full_block_count * block_size
        integral_sum = _tree_add(
            integral_sum,
            reduce_values(
                gammas[:, tail_start:],
                gammadashs[:, tail_start:],
            ),
        )
    return average_sum(integral_sum)


def _one_point_dense_B_and_dB(
    x, gammas, gammadashs, currents, *, quadrature_block_size
):
    B_integral, dB_integral = _quadrature_block_B_and_dB_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
    )
    scale = _float64_scalar(currents, _MU0_OVER_4PI)
    B = scale * jnp.einsum(
        "c,cj->j",
        currents,
        B_integral,
        precision=lax.Precision.HIGHEST,
    )
    dB = scale * jnp.einsum(
        "c,cjl->jl",
        currents,
        dB_integral,
        precision=lax.Precision.HIGHEST,
    )
    return B, dB


# ── Kernel factory ────────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _make_kernel(
    integrand_key,
    diff_mode,
    coil_cs,
    quad_bs,
    point_cs,
):
    """Build and JIT-compile a Biot-Savart kernel for the given tuning config.

    All tiling parameters are captured in closures — callers never thread them.
    ``lru_cache`` ensures the same config returns the same compiled function.

    Cache keyed on ``(integrand_key, diff_mode, coil_cs, quad_bs, point_cs)``.
    JAX owns per-backend executable caching below the
    Python closure, so backend identity is not part of this LRU key.
    """
    integrand = _INTEGRANDS[integrand_key]

    def one_point(x, gammas, gammadashs, currents):
        return _coil_chunk_reduce(
            gammas,
            gammadashs,
            currents,
            chunk_size=coil_cs,
            zero=_zeros((3,), dtype=jnp.float64),
            reduce_chunk=lambda cg, cgd, cc: _one_point_dense(
                x,
                cg,
                cgd,
                cc,
                integrand=integrand,
                quadrature_block_size=quad_bs,
            ),
        )

    if diff_mode is _DiffMode.VALUE:
        per_point = one_point
    elif diff_mode is _DiffMode.JACOBIAN:

        def jacobian_point(x, gammas, gammadashs, currents):
            return jnp.swapaxes(
                jax.jacfwd(one_point, argnums=0)(x, gammas, gammadashs, currents),
                -1,
                -2,
            )
        per_point = jacobian_point

    elif diff_mode is _DiffMode.HESSIAN:

        def hessian_point(x, gammas, gammadashs, currents):
            # Transpose (1,2,0): raw jacfwd² yields [component, d1, d2],
            # upstream convention is [d1, d2, component].
            # jacfwd² (not jacrev∘jacfwd) preserves exact current-linearity.
            return jnp.transpose(
                jax.jacfwd(jax.jacfwd(one_point, argnums=0), argnums=0)(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                ),
                (1, 2, 0),
            )
        per_point = hessian_point

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN and integrand_key is _Integrand.B:

        def B_and_jacobian_point(x, gammas, gammadashs, currents):
            return _coil_chunk_reduce(
                gammas,
                gammadashs,
                currents,
                chunk_size=coil_cs,
                zero=(
                    _zeros((3,), dtype=jnp.float64),
                    _zeros((3, 3), dtype=jnp.float64),
                ),
                reduce_chunk=lambda cg, cgd, cc: _one_point_dense_B_and_dB(
                    x,
                    cg,
                    cgd,
                    cc,
                    quadrature_block_size=quad_bs,
                ),
            )
        per_point = B_and_jacobian_point

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN:

        def value_and_jacobian_point(x, gammas, gammadashs, currents):
            f = lambda xx: one_point(xx, gammas, gammadashs, currents)
            primals, tangents_fn = jax.linearize(f, x)
            basis = _eye(3, dtype=jnp.float64)
            return primals, jax.vmap(tangents_fn, in_axes=(0,))(basis)
        per_point = value_and_jacobian_point

    kernel_point_cs = (
        hessian_point_chunk_size(point_cs)
        if diff_mode is _DiffMode.HESSIAN
        else point_cs
    )

    def kernel(points, gammas, gammadashs, currents):
        def chunk_fn(chunk_points):
            return jax.vmap(
                lambda x: per_point(x, gammas, gammadashs, currents),
                in_axes=(0,),
            )(chunk_points)

        return _point_chunk_reduce(points, chunk_fn, kernel_point_cs)

    kernel.__name__ = "biotsavart_forward"
    return jax.jit(kernel)


def _get_kernel(integrand_key, diff_mode):
    """Read tuning config and return the cached JIT-compiled kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_kernel(integrand_key, diff_mode, coil_cs, quad_bs, point_cs)


@lru_cache(maxsize=64)
def _make_B_vjp_kernel(coil_cs, quad_bs, point_cs):
    """Build the tuning-keyed compiled VJP kernel for ``biot_savart_B``.

    ``biot_savart_B_vjp`` differentiates through the forward field kernel, so it
    must be rebuilt when the backend tuning changes in the same process.
    Keying this factory on the same tuning tuple as the forward kernels keeps
    the backend/performance contract consistent across both paths.
    """
    forward_kernel = _make_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        coil_cs,
        quad_bs,
        point_cs,
    )

    def kernel(points, v, gammas, gammadashs, currents):
        def fwd(group_gammas, group_gammadashs, group_currents):
            return forward_kernel(
                points,
                group_gammas,
                group_gammadashs,
                group_currents,
            )

        _, pullback = jax.vjp(fwd, gammas, gammadashs, currents)
        return pullback(v)

    kernel.__name__ = "biotsavart_vjp"
    return jax.jit(kernel)


def _get_B_vjp_kernel():
    """Read tuning config and return the cached JIT-compiled VJP kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_B_vjp_kernel(coil_cs, quad_bs, point_cs)


@lru_cache(maxsize=64)
def _make_d2B_vjp_kernel(point_chunk_size):
    """Build the compiled closed-form gradgradB pullback, tiled over points.

    Each tile of ``point_chunk_size`` points adds its per-node sums of
    :func:`_hessian_vjp_tile_sums`; nonpositive sizes evaluate all points in
    one tile.  Live intermediates scale with the tile times ``C * Q``.
    """

    def kernel(points, vgradgrad, gammas, gammadashs, currents):
        point_count = points.shape[0]
        if point_count == 0:
            return jax.tree.map(jnp.zeros_like, (gammas, gammadashs, currents))
        if point_chunk_size <= 0 or point_count <= point_chunk_size:
            grad_sum, c_sum = _hessian_vjp_tile_sums(points, vgradgrad, gammas, gammadashs)
        else:
            full_chunk_count, tail_size = divmod(point_count, point_chunk_size)

            def body(chunk_index, sums):
                start = chunk_index * point_chunk_size
                return _tree_add(sums, _hessian_vjp_tile_sums(
                    _slice_point_chunk(points, start, point_chunk_size),
                    _slice_point_chunk(vgradgrad, start, point_chunk_size),
                    gammas, gammadashs,
                ))

            zero = (jnp.zeros_like(gammas), jnp.zeros_like(gammas))
            grad_sum, c_sum = lax.fori_loop(0, full_chunk_count, body, zero)
            if tail_size:
                tail_start = full_chunk_count * point_chunk_size
                grad_sum, c_sum = _tree_add((grad_sum, c_sum), _hessian_vjp_tile_sums(
                    points[tail_start:], vgradgrad[tail_start:], gammas, gammadashs,
                ))
        scale = _float64_scalar(currents, _MU0_OVER_4PI) / gammas.shape[1]
        coil_scale = (scale * currents)[:, None, None]
        current_cotangent = scale * _pairwise_sum_axis(jnp.sum(gammadashs * c_sum, axis=-1), axis=1)
        return -coil_scale * grad_sum, coil_scale * c_sum, current_cotangent

    kernel.__name__ = "biotsavart_d2B_vjp"
    return jax.jit(kernel)


def invalidate_kernel_cache() -> None:
    """Drop all cached JIT-compiled Biot-Savart kernels and tuning config.

    Both the forward and the pullback kernel factories are cleared. Call after
    overriding ``_read_tuning_config`` (e.g. via ``monkeypatch``) to ensure
    the next ``biot_savart_*`` call rebuilds with the new config.
    """
    _make_kernel.cache_clear()
    _make_B_vjp_kernel.cache_clear()
    _make_d2B_vjp_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def _apply_forward_kernel(
    integrand_key,
    diff_mode,
    points,
    gammas,
    gammadashs,
    currents,
):
    """Apply one cached field kernel under the canonical forward trace scope."""

    inputs = snapshot_host_tree((points, gammas, gammadashs, currents))
    return _get_kernel(integrand_key, diff_mode)(*inputs)


def biot_savart_B(points, gammas, gammadashs, currents):
    """Evaluate the Biot-Savart B kernel for one rectangular coil group.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Magnetic field in tesla, shape (P, 3), Cartesian component
            last.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    """First spatial gradient of the Biot-Savart magnetic field.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla per meter; result[p, j, l] =
            partial_j B_l at point p.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_d2B_by_dXdX(points, gammas, gammadashs, currents):
    """Return the dense Hessian of ``B`` with shape ``(npoints, 3, 3, 3)``.

    This is an opt-in diagnostics kernel: each point materializes three dense
    derivative planes per field component, so callers should prefer
    ``biot_savart_B_and_dB`` unless second point derivatives are required.
    The pre-reduction integrand is large: for example, ``P_chunk=512``,
    ``C=16``, ``Q=128`` materializes roughly 226 MB (216 MiB) of Hessian
    intermediates before quadrature reduction.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter squared; result[p,
            i, j, l] = partial_i partial_j B_l.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.HESSIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_d2B_by_dXdX_vjp(points, vgradgrad, gammas, gammadashs, currents):
    """Pull back a magnetic-field Hessian seed to coil geometry and currents.

    Contracts ``vgradgrad[p, i, j, c]`` with ``d_i d_j B_c(points[p])``.
    The seed need not be symmetric. Each point tile evaluates a closed-form
    contraction of the third spatial derivatives of the integrand, so no
    Hessian is formed; points are held fixed.

    Args:
        points (jax.Array or numpy.ndarray): Cartesian positions of shape
            ``(npoints, 3)``, in meters.
        vgradgrad (jax.Array or numpy.ndarray): Hessian cotangent of shape
            ``(npoints, 3, 3, 3)`` in ``[point, d1, d2, component]`` order.
            For a scalar objective with units U, the seed has units U m²/T.
        gammas (jax.Array or numpy.ndarray): Coil positions of shape
            ``(ncoils, nquad, 3)``, in meters.
        gammadashs (jax.Array or numpy.ndarray): Coil tangents of shape
            ``(ncoils, nquad, 3)``, in meters per unit dimensionless curve
            parameter; quadrature uses uniformly spaced nodes on [0, 1).
        currents (jax.Array or numpy.ndarray): Coil currents of shape
            ``(ncoils,)``, in amperes.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Cotangents of ``gammas``,
            ``gammadashs`` and ``currents``, with their respective input shapes
            ``(ncoils, nquad, 3)``, ``(ncoils, nquad, 3)`` and ``(ncoils,)``.
            For an objective in U, geometry cotangents have units U/m and current
            cotangents have units U/A.
    """
    inputs = snapshot_host_tree((points, vgradgrad, gammas, gammadashs, currents))
    tuning = get_field_kernel_tuning()
    return _make_d2B_vjp_kernel(tuning.hessian_vjp_point_chunk_size)(*inputs)


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    """Return ``(B, dB_by_dX)`` for the Biot-Savart field.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        tuple[jax.Array, jax.Array]: B of shape (P, 3) in tesla and its
            Jacobian of shape (P, 3, 3) in tesla per meter, with derivative
            direction before field component.
    """
    return _apply_forward_kernel(
        _Integrand.B,
        _DiffMode.VALUE_AND_JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    """Evaluate the Biot-Savart A kernel for one rectangular coil group.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Vector potential in tesla meters, shape (P, 3), Cartesian
            component last.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.VALUE,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    """First spatial gradient of the Biot-Savart vector potential.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3), in tesla; result[p, j, l] = partial_j A_l
            at point p.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.JACOBIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_d2A_by_dXdX(points, gammas, gammadashs, currents):
    """Evaluate the Biot-Savart d2A_by_dXdX kernel for one rectangular coil group.

    Uniform quadrature averages over Q samples of the normalized curve parameter, using mu0/(4*pi) = 1e-7. Evaluation on a filament retains the physical singularity.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        jax.Array: Shape (P, 3, 3, 3), in tesla per meter; result[p, i, j, l]
            = partial_i partial_j A_l.
    """
    return _apply_forward_kernel(
        _Integrand.A,
        _DiffMode.HESSIAN,
        points,
        gammas,
        gammadashs,
        currents,
    )


def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """VJP of ``biot_savart_B`` w.r.t. coil data.

    Returns raw cotangents for ``(gammas, gammadashs, currents)``. The CPU
    ``BiotSavart.B_vjp`` wrapper instead pushes geometry and current
    cotangents through each ``Coil.vjp(...)`` and returns a ``Derivative``
    over coil/current dofs, so downstream comparisons must combine the raw
    JAX leaves through the same coil mapping.

    Uses a dedicated tuning-keyed kernel factory so backend mode and chunking
    changes rebuild the compiled closure in the same process, matching the
    cache invalidation behavior of the forward ``biot_savart_B`` kernels.

    Args:
        points (array-like): Cartesian evaluation points, shape (P, 3), in meters.
        v (array-like): B cotangent, shape (P, 3), contracting the output as sum(v * B);
            units depend on the scalar objective.
        gammas (array-like): Coil positions, shape (C, Q, 3), in meters.
        gammadashs (array-like): Coil derivatives with respect to the normalized
            parameter, shape (C, Q, 3), in meters.
        currents (array-like): Coil currents, shape (C,), in amperes.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Cotangents for positions,
            tangents and currents, shapes (C, Q, 3), (C, Q, 3), (C,). Units are
            those of the contracted scalar per meter or per ampere; points are
            held fixed.
    """
    inputs = snapshot_host_tree((points, v, gammas, gammadashs, currents))
    return _get_B_vjp_kernel()(*inputs)


# ── Grouped coil utilities ───────────────────────────────────────────


def _coil_entry_sequence(values: object) -> tuple[jax.Array, ...]:
    if isinstance(values, tuple):
        return cast(tuple[jax.Array, ...], values)
    if isinstance(values, list):
        return cast(tuple[jax.Array, ...], tuple(values))
    return _axis0_entries(values)


def group_coil_data(
    gammas_list,
    gammadashs_list,
    currents_list,
    *,
    use_compute_dtype: bool = True,
):
    """Batch coils by quadrature count while preserving original index mappings.

    Args:
        gammas_list (Sequence[array-like]): Per-coil positions, each shape (Q_i, 3), in
            meters; a rectangular array of shape (C, Q, 3) is also accepted.
        gammadashs_list (Sequence[array-like]): Matching parameter derivatives, each
            shape (Q_i, 3), in meters.
        currents_list (Sequence[scalar]): Matching coil currents in amperes, each scalar
            shape ().
        use_compute_dtype (bool): Convert groups to compute precision when True
            (default), runtime precision otherwise.

    Returns:
        list[tuple]: One (gammas, gammadashs, currents, indices) tuple per
            quadrature count, with shapes (C_g, Q_g, 3), (C_g, Q_g, 3), (C_g,) and
            original index lists. Groups are ordered by their first coil index.
    """
    gamma_entries = _coil_entry_sequence(gammas_list)
    gammadash_entries = _coil_entry_sequence(gammadashs_list)
    current_entries = _coil_entry_sequence(currents_list)
    indices_by_nquad = {}
    for i, gamma in enumerate(gamma_entries):
        indices_by_nquad.setdefault(gamma.shape[0], []).append(i)

    array_for_group = _as_compute_array if use_compute_dtype else _as_jax_float64
    groups = []
    for indices in sorted(indices_by_nquad.values(), key=lambda group: group[0]):
        groups.append(
            (
                jnp.stack([array_for_group(gamma_entries[i]) for i in indices]),
                jnp.stack([array_for_group(gammadash_entries[i]) for i in indices]),
                jnp.stack([array_for_group(current_entries[i]) for i in indices]),
                indices,
            )
        )
    return groups
