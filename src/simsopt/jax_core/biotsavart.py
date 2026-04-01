"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
replace the C++ ``simsoptpp.BiotSavart`` kernel for GPU execution.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from ..backend import (
    get_coil_chunk_size,
    get_point_chunk_size,
    get_quadrature_block_size,
)

__all__ = [
    "biot_savart_B",
    "biot_savart_B_vjp",
    "biot_savart_dB_by_dX",
    "biot_savart_B_and_dB",
    "biot_savart_A",
    "biot_savart_dA_by_dX",
    "group_coil_data",
    "grouped_biot_savart_B",
    "grouped_biot_savart_A",
]

_MU0_OVER_4PI = 1e-7


def _coil_chunk_size() -> int:
    return get_coil_chunk_size()


def _quadrature_block_size() -> int:
    return get_quadrature_block_size()


def _point_chunk_size() -> int:
    return get_point_chunk_size()


def _slice_coil_chunk(array, start: int, chunk_size: int):
    trailing_shape = tuple(array.shape[1:])
    return lax.dynamic_slice(
        array,
        (start,) + (0,) * len(trailing_shape),
        (chunk_size,) + trailing_shape,
    )


def _slice_quadrature_block(array, start: int, block_size: int):
    return lax.dynamic_slice(
        array,
        (0, start, 0),
        (array.shape[0], block_size, array.shape[2]),
    )


def _slice_point_chunk(points: object, start: int, chunk_size: int):
    return lax.dynamic_slice(
        points,
        (start, 0),
        (chunk_size, points.shape[1]),
    )


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_dynamic_update(prefix_tree, chunk_tree, start_index: int):
    return jax.tree_util.tree_map(
        lambda acc, update: lax.dynamic_update_slice(
            acc,
            update,
            (start_index,) + (0,) * (acc.ndim - 1),
        ),
        prefix_tree,
        chunk_tree,
    )


def _tree_trim(prefix_tree, size: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[:size], prefix_tree)


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(
            (prefix_size,) + tuple(leaf.shape[1:]),
            dtype=leaf.dtype,
        ),
        reference_tree,
    )


def _coil_chunk_reduce(
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    zero,
    reduce_chunk,
):
    coil_count = int(currents.shape[0])
    if coil_count == 0:
        return zero
    if chunk_size <= 0 or coil_count <= chunk_size:
        return reduce_chunk(gammas, gammadashs, currents)

    chunk_count = (coil_count + chunk_size - 1) // chunk_size
    padded_coil_count = chunk_count * chunk_size
    pad_width = ((0, padded_coil_count - coil_count), (0, 0), (0, 0))
    padded_gammas = jnp.pad(gammas, pad_width)
    padded_gammadashs = jnp.pad(gammadashs, pad_width)
    padded_currents = jnp.pad(currents, ((0, padded_coil_count - coil_count),))

    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_gammas = _slice_coil_chunk(padded_gammas, start, chunk_size)
        chunk_gammadashs = _slice_coil_chunk(padded_gammadashs, start, chunk_size)
        chunk_currents = lax.dynamic_slice(
            padded_currents,
            (start,),
            (chunk_size,),
        )
        return acc + reduce_chunk(chunk_gammas, chunk_gammadashs, chunk_currents)

    return lax.fori_loop(0, chunk_count, body, zero)


def _quadrature_block_integral(
    x,
    gammas,
    gammadashs,
    *,
    block_size: int,
    integrand,
):
    quadrature_count = int(gammas.shape[1])
    if block_size <= 0 or quadrature_count <= block_size:
        return jnp.mean(integrand(x, gammas, gammadashs), axis=1)

    block_count = (quadrature_count + block_size - 1) // block_size
    padded_quadrature_count = block_count * block_size
    pad_width = ((0, 0), (0, padded_quadrature_count - quadrature_count), (0, 0))
    padded_gammas = jnp.pad(gammas, pad_width)
    padded_gammadashs = jnp.pad(gammadashs, pad_width)
    zero = jnp.zeros((gammas.shape[0], 3), dtype=jnp.float64)

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(padded_gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            padded_gammadashs,
            start,
            block_size,
        )
        block_integrand = integrand(x, block_gammas, block_gammadashs)
        return acc + jnp.sum(block_integrand, axis=1)

    integral_sum = lax.fori_loop(0, block_count, body, zero)
    return integral_sum / quadrature_count


def _point_chunk_reduce(points: object, chunk_kernel):
    point_count = int(points.shape[0])
    chunk_size = _point_chunk_size()
    if point_count == 0 or chunk_size <= 0 or point_count <= chunk_size:
        return chunk_kernel(points)

    chunk_count = (point_count + chunk_size - 1) // chunk_size
    padded_point_count = chunk_count * chunk_size
    padded_points = jnp.pad(
        points,
        ((0, padded_point_count - point_count), (0, 0)),
    )
    first_chunk_points = _slice_point_chunk(padded_points, 0, chunk_size)
    first_result = chunk_kernel(first_chunk_points)
    padded_result = _tree_dynamic_update(
        _tree_zeros_like_prefix(first_result, padded_point_count),
        first_result,
        0,
    )

    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_points = _slice_point_chunk(padded_points, start, chunk_size)
        chunk_result = chunk_kernel(chunk_points)
        return _tree_dynamic_update(acc, chunk_result, start)

    padded_result = lax.fori_loop(1, chunk_count, body, padded_result)
    return _tree_trim(padded_result, point_count)


def _biot_savart_B_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv3 = jnp.where(r2 > 0, safe_r2 ** (-1.5), 0.0)
    cross = jnp.cross(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_one_point_dense(
    x,
    gammas,
    gammadashs,
    currents,
    *,
    quadrature_block_size: int = 0,
):
    integral = _quadrature_block_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
        integrand=_biot_savart_B_integrand,
    )
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


def _chunked_one_point(
    dense_kernel,
    x,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _coil_chunk_reduce(
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        zero=jnp.zeros((3,), dtype=jnp.float64),
        reduce_chunk=lambda chunk_gammas,
        chunk_gammadashs,
        chunk_currents: dense_kernel(
            x,
            chunk_gammas,
            chunk_gammadashs,
            chunk_currents,
            quadrature_block_size=quadrature_block_size,
        ),
    )


def _biot_savart_one_point(
    x,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _chunked_one_point(
        _biot_savart_one_point_dense,
        x,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


def _pointwise_kernel(
    points,
    point_kernel,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    def chunk_kernel(chunk_points):
        return jax.vmap(
            lambda x: point_kernel(
                x,
                gammas,
                gammadashs,
                currents,
                chunk_size=chunk_size,
                quadrature_block_size=quadrature_block_size,
            ),
            in_axes=0,
        )(chunk_points)

    return _point_chunk_reduce(points, chunk_kernel)


def _pointwise_jacobian(
    points,
    point_kernel,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    jac_fn = jax.jacfwd(point_kernel, argnums=0)

    def chunk_kernel(chunk_points):
        raw = jax.vmap(
            lambda x: jac_fn(
                x,
                gammas,
                gammadashs,
                currents,
                chunk_size=chunk_size,
                quadrature_block_size=quadrature_block_size,
            ),
            in_axes=0,
        )(chunk_points)
        return jnp.swapaxes(raw, -1, -2)

    return _point_chunk_reduce(points, chunk_kernel)


@partial(jax.jit, static_argnames=("chunk_size", "quadrature_block_size"))
def _biot_savart_B_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _pointwise_kernel(
        points,
        _biot_savart_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


def biot_savart_B(points, gammas, gammadashs, currents):
    return _biot_savart_B_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
        quadrature_block_size=_quadrature_block_size(),
    )


@jax.jit
def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    def fwd(group_gammas, group_gammadashs, group_currents):
        return biot_savart_B(points, group_gammas, group_gammadashs, group_currents)

    _, pullback = jax.vjp(fwd, gammas, gammadashs, currents)
    return pullback(v)


@partial(jax.jit, static_argnames=("chunk_size", "quadrature_block_size"))
def _biot_savart_dB_by_dX_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _pointwise_jacobian(
        points,
        _biot_savart_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    return _biot_savart_dB_by_dX_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
        quadrature_block_size=_quadrature_block_size(),
    )


@partial(jax.jit, static_argnames=("chunk_size", "quadrature_block_size"))
def _biot_savart_B_and_dB_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    def _val_and_jac(x):
        f = lambda xx: _biot_savart_one_point(
            xx,
            gammas,
            gammadashs,
            currents,
            chunk_size=chunk_size,
            quadrature_block_size=quadrature_block_size,
        )
        primals, tangents_fn = jax.linearize(f, x)
        return primals, jax.vmap(tangents_fn)(jnp.eye(3))

    return _point_chunk_reduce(
        points,
        lambda chunk_points: jax.vmap(_val_and_jac)(chunk_points),
    )


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    B, dB_dX = _biot_savart_B_and_dB_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
        quadrature_block_size=_quadrature_block_size(),
    )
    return B, dB_dX


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv = jnp.where(r2 > 0, safe_r2 ** (-0.5), 0.0)
    return gammadashs * r_inv[..., None]


def _biot_savart_A_one_point_dense(
    x,
    gammas,
    gammadashs,
    currents,
    *,
    quadrature_block_size: int = 0,
):
    integral = _quadrature_block_integral(
        x,
        gammas,
        gammadashs,
        block_size=quadrature_block_size,
        integrand=_biot_savart_A_integrand,
    )
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


def _biot_savart_A_one_point(
    x,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _chunked_one_point(
        _biot_savart_A_one_point_dense,
        x,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


@partial(jax.jit, static_argnames=("chunk_size", "quadrature_block_size"))
def _biot_savart_A_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _pointwise_kernel(
        points,
        _biot_savart_A_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    return _biot_savart_A_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
        quadrature_block_size=_quadrature_block_size(),
    )


@partial(jax.jit, static_argnames=("chunk_size", "quadrature_block_size"))
def _biot_savart_dA_by_dX_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    quadrature_block_size: int,
):
    return _pointwise_jacobian(
        points,
        _biot_savart_A_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
        quadrature_block_size=quadrature_block_size,
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    return _biot_savart_dA_by_dX_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
        quadrature_block_size=_quadrature_block_size(),
    )


def group_coil_data(gammas_list, gammadashs_list, currents_list):
    by_nquad = {}
    for i, gamma in enumerate(gammas_list):
        by_nquad.setdefault(gamma.shape[0], []).append(i)

    groups = []
    for indices in by_nquad.values():
        groups.append(
            (
                jnp.stack(
                    [jnp.asarray(gammas_list[i], dtype=jnp.float64) for i in indices]
                ),
                jnp.stack(
                    [
                        jnp.asarray(gammadashs_list[i], dtype=jnp.float64)
                        for i in indices
                    ]
                ),
                jnp.stack(
                    [jnp.asarray(currents_list[i], dtype=jnp.float64) for i in indices]
                ),
                indices,
            )
        )
    return groups


def grouped_biot_savart_B(points, coil_arrays):
    g0, gd0, c0 = coil_arrays[0]
    result = biot_savart_B(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + biot_savart_B(points, gammas, gammadashs, currents)
    return result


def grouped_biot_savart_A(points, coil_arrays):
    g0, gd0, c0 = coil_arrays[0]
    result = biot_savart_A(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + biot_savart_A(points, gammas, gammadashs, currents)
    return result
