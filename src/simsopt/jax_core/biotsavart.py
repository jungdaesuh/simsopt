"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
replace the C++ ``simsoptpp.BiotSavart`` kernel for GPU execution.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from functools import partial
import os

import jax
from jax import lax
import jax.numpy as jnp

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
_MODE_ENV = "SIMSOPT_BACKEND_MODE"
_COIL_CHUNK_SIZE_BY_MODE = {
    "native_cpu": 0,
    "jax_cpu_parity": 16,
    "jax_gpu_parity": 16,
    "jax_gpu_fast": 64,
}


def _coil_chunk_size() -> int:
    mode = os.environ.get(_MODE_ENV, "native_cpu")
    return int(_COIL_CHUNK_SIZE_BY_MODE.get(mode, 0))


def _slice_coil_chunk(array, start: int, chunk_size: int):
    trailing_shape = tuple(array.shape[1:])
    return lax.dynamic_slice(
        array,
        (start,) + (0,) * len(trailing_shape),
        (chunk_size,) + trailing_shape,
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


def _biot_savart_one_point_dense(x, gammas, gammadashs, currents):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv3 = jnp.where(r2 > 0, safe_r2 ** (-1.5), 0.0)
    cross = jnp.cross(diff, gammadashs)
    integrand = cross * r_inv3[..., None]
    integral = jnp.mean(integrand, axis=1)
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


def _chunked_one_point(
    dense_kernel,
    x,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
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
        ),
    )


def _biot_savart_one_point(x, gammas, gammadashs, currents, *, chunk_size: int):
    return _chunked_one_point(
        _biot_savart_one_point_dense,
        x,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


def _pointwise_kernel(
    points, point_kernel, gammas, gammadashs, currents, *, chunk_size: int
):
    return jax.vmap(
        lambda x: point_kernel(
            x,
            gammas,
            gammadashs,
            currents,
            chunk_size=chunk_size,
        ),
        in_axes=0,
    )(points)


def _pointwise_jacobian(
    points,
    point_kernel,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
):
    jac_fn = jax.jacfwd(point_kernel, argnums=0)
    raw = jax.vmap(
        lambda x: jac_fn(
            x,
            gammas,
            gammadashs,
            currents,
            chunk_size=chunk_size,
        ),
        in_axes=0,
    )(points)
    return jnp.swapaxes(raw, -1, -2)


@partial(jax.jit, static_argnames=("chunk_size",))
def _biot_savart_B_impl(points, gammas, gammadashs, currents, *, chunk_size: int):
    return _pointwise_kernel(
        points,
        _biot_savart_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


def biot_savart_B(points, gammas, gammadashs, currents):
    return _biot_savart_B_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )


@jax.jit
def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    def fwd(group_gammas, group_gammadashs, group_currents):
        return biot_savart_B(points, group_gammas, group_gammadashs, group_currents)

    _, pullback = jax.vjp(fwd, gammas, gammadashs, currents)
    return pullback(v)


@partial(jax.jit, static_argnames=("chunk_size",))
def _biot_savart_dB_by_dX_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
):
    return _pointwise_jacobian(
        points,
        _biot_savart_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    return _biot_savart_dB_by_dX_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def _biot_savart_B_and_dB_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
):
    def _val_and_jac(x):
        f = lambda xx: _biot_savart_one_point(
            xx,
            gammas,
            gammadashs,
            currents,
            chunk_size=chunk_size,
        )
        primals, tangents_fn = jax.linearize(f, x)
        return primals, jax.vmap(tangents_fn)(jnp.eye(3))

    return jax.vmap(_val_and_jac)(points)


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    B, dB_dX = _biot_savart_B_and_dB_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )
    return B, dB_dX


def _biot_savart_A_one_point_dense(x, gammas, gammadashs, currents):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv = jnp.where(r2 > 0, safe_r2 ** (-0.5), 0.0)
    integrand = gammadashs * r_inv[..., None]
    integral = jnp.mean(integrand, axis=1)
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


def _biot_savart_A_one_point(x, gammas, gammadashs, currents, *, chunk_size: int):
    return _chunked_one_point(
        _biot_savart_A_one_point_dense,
        x,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def _biot_savart_A_impl(points, gammas, gammadashs, currents, *, chunk_size: int):
    return _pointwise_kernel(
        points,
        _biot_savart_A_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    return _biot_savart_A_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def _biot_savart_dA_by_dX_impl(
    points,
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
):
    return _pointwise_jacobian(
        points,
        _biot_savart_A_one_point,
        gammas,
        gammadashs,
        currents,
        chunk_size=chunk_size,
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    return _biot_savart_dA_by_dX_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
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
