"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
replace the C++ ``simsoptpp.BiotSavart`` kernel for GPU execution.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from functools import lru_cache

import jax
from jax import lax
import jax.numpy as jnp

from ..backend import (
    get_field_kernel_tuning,
    get_point_chunk_size,
)
from ..backend.runtime import register_backend_cache_clear

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
    "invalidate_kernel_cache",
]

_MU0_OVER_4PI = 1e-7


# ── Config reader (monkeypatchable by tests) ──────────────────────────


def _read_tuning_config() -> tuple:
    """Return ``(coil_chunk_size, quadrature_block_size, point_chunk_size)``.

    Single indirection point for all tuning knobs consumed by the kernel
    factory.  Tests override this one function (+ ``invalidate_kernel_cache``)
    instead of patching three separate stubs.

    Resolves the backend config once to avoid repeated mode/policy lookups.
    """
    fkt = get_field_kernel_tuning()
    return fkt.coil_chunk_size, fkt.quadrature_block_size, get_point_chunk_size()


# ── Array slicing primitives ──────────────────────────────────────────


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


# ── Tree utilities ────────────────────────────────────────────────────


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


def _tree_concatenate(left, right):
    return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=0), left, right
    )


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(
            (prefix_size,) + tuple(leaf.shape[1:]),
            dtype=leaf.dtype,
        ),
        reference_tree,
    )


# ── Tiling primitives ────────────────────────────────────────────────


def _two_chunk_sum(first_chunk, second_chunk, reduce_chunk):
    return reduce_chunk(*first_chunk) + reduce_chunk(*second_chunk)


def _coil_chunk_reduce(
    gammas,
    gammadashs,
    currents,
    *,
    chunk_size: int,
    zero,
    reduce_chunk,
):
    coil_count = currents.shape[0]
    if coil_count == 0:
        return zero
    if chunk_size <= 0 or coil_count <= chunk_size:
        return reduce_chunk(gammas, gammadashs, currents)

    chunk_count = (coil_count + chunk_size - 1) // chunk_size
    if chunk_count == 2:
        return _two_chunk_sum(
            (
                gammas[:chunk_size],
                gammadashs[:chunk_size],
                currents[:chunk_size],
            ),
            (
                gammas[chunk_size:],
                gammadashs[chunk_size:],
                currents[chunk_size:],
            ),
            reduce_chunk,
        )

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
    quadrature_count = gammas.shape[1]
    if block_size <= 0 or quadrature_count <= block_size:
        return jnp.mean(integrand(x, gammas, gammadashs), axis=1)

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
        return (
            _two_chunk_sum(
                (
                    gammas[:, :block_size, :],
                    gammadashs[:, :block_size, :],
                ),
                (
                    gammas[:, block_size:, :],
                    gammadashs[:, block_size:, :],
                ),
                lambda gg, ggd: jnp.sum(integrand(x, gg, ggd), axis=1),
            )
            / quadrature_count
        )

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


def _point_chunk_reduce(points, chunk_kernel, chunk_size):
    """Evaluate *chunk_kernel* over *points* with optional point-axis tiling.

    Three-arg signature: ``chunk_size`` is passed explicitly by the kernel
    factory (not read from module state) so that the value is part of the
    ``lru_cache`` key and triggers recompilation when it changes.
    """
    point_count = points.shape[0]
    if point_count == 0 or chunk_size <= 0 or point_count <= chunk_size:
        return chunk_kernel(points)

    chunk_count = (point_count + chunk_size - 1) // chunk_size
    if chunk_count == 2:
        return _tree_concatenate(
            chunk_kernel(points[:chunk_size]),
            chunk_kernel(points[chunk_size:]),
        )

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


# ── Physics integrands ────────────────────────────────────────────────


def _biot_savart_B_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv3 = jnp.where(r2 > 0, safe_r2 ** (-1.5), 0.0)
    cross = jnp.cross(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv = jnp.where(r2 > 0, safe_r2 ** (-0.5), 0.0)
    return gammadashs * r_inv[..., None]


# ── Dense one-point kernels ──────────────────────────────────────────


_INTEGRAND_B = "B"
_INTEGRAND_A = "A"

_DIFF_VALUE = "value"
_DIFF_JACOBIAN = "jacobian"
_DIFF_VALUE_AND_JACOBIAN = "value_and_jacobian"

_INTEGRANDS = {
    _INTEGRAND_B: _biot_savart_B_integrand,
    _INTEGRAND_A: _biot_savart_A_integrand,
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
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


# ── Kernel factory ────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _make_kernel(integrand_key, diff_mode, coil_cs, quad_bs, point_cs):
    """Build and JIT-compile a Biot-Savart kernel for the given tuning config.

    All tiling parameters are captured in closures — callers never thread them.
    ``lru_cache`` ensures the same config returns the same compiled function.

    Cache keyed on ``(integrand_key, diff_mode, coil_cs, quad_bs, point_cs)``.
    A config change that produces different int values naturally creates a new
    cache entry.  Call ``_make_kernel.cache_clear()`` if you need to force
    recompilation (e.g. after hot-patching an integrand function in tests).
    """
    integrand = _INTEGRANDS.get(integrand_key)
    if integrand is None:
        raise ValueError(
            f"Unknown integrand_key: {integrand_key!r}. Expected one of {set(_INTEGRANDS)}"
        )

    def one_point(x, gammas, gammadashs, currents):
        return _coil_chunk_reduce(
            gammas,
            gammadashs,
            currents,
            chunk_size=coil_cs,
            zero=jnp.zeros((3,), dtype=jnp.float64),
            reduce_chunk=lambda cg, cgd, cc: _one_point_dense(
                x,
                cg,
                cgd,
                cc,
                integrand=integrand,
                quadrature_block_size=quad_bs,
            ),
        )

    if diff_mode == _DIFF_VALUE:
        per_point = one_point
    elif diff_mode == _DIFF_JACOBIAN:

        def per_point(x, gammas, gammadashs, currents):
            return jnp.swapaxes(
                jax.jacfwd(one_point, argnums=0)(x, gammas, gammadashs, currents),
                -1,
                -2,
            )

    elif diff_mode == _DIFF_VALUE_AND_JACOBIAN:

        def per_point(x, gammas, gammadashs, currents):
            f = lambda xx: one_point(xx, gammas, gammadashs, currents)
            primals, tangents_fn = jax.linearize(f, x)
            return primals, jax.vmap(tangents_fn)(jnp.eye(3))

    else:
        raise ValueError(f"Unknown diff_mode: {diff_mode!r}")

    @jax.jit
    def kernel(points, gammas, gammadashs, currents):
        def chunk_fn(chunk_points):
            return jax.vmap(
                lambda x: per_point(x, gammas, gammadashs, currents),
            )(chunk_points)

        return _point_chunk_reduce(points, chunk_fn, point_cs)

    return kernel


def _get_kernel(integrand_key, diff_mode):
    """Read tuning config and return the cached JIT-compiled kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_kernel(integrand_key, diff_mode, coil_cs, quad_bs, point_cs)


def invalidate_kernel_cache() -> None:
    """Drop all cached JIT-compiled Biot-Savart kernels and tuning config.

    Call after overriding ``_read_tuning_config`` (e.g. via ``monkeypatch``)
    to ensure the next ``biot_savart_*`` call rebuilds with the new config.
    """
    _make_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def biot_savart_B(points, gammas, gammadashs, currents):
    return _get_kernel(_INTEGRAND_B, _DIFF_VALUE)(points, gammas, gammadashs, currents)


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    return _get_kernel(_INTEGRAND_B, _DIFF_JACOBIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    return _get_kernel(_INTEGRAND_B, _DIFF_VALUE_AND_JACOBIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    return _get_kernel(_INTEGRAND_A, _DIFF_VALUE)(points, gammas, gammadashs, currents)


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    return _get_kernel(_INTEGRAND_A, _DIFF_JACOBIAN)(
        points, gammas, gammadashs, currents
    )


@jax.jit
def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """VJP of ``biot_savart_B`` w.r.t. coil data.

    Uses bare ``@jax.jit`` (not the kernel factory) because the VJP
    structure differs from a plain field eval.  The inner
    ``biot_savart_B`` call goes through ``_get_kernel`` at trace time,
    so tuning config IS respected on first compilation.  However, a
    mid-process config change will not invalidate the outer JIT cache —
    acceptable because the backend mode is set once at startup.
    """

    def fwd(group_gammas, group_gammadashs, group_currents):
        return biot_savart_B(points, group_gammas, group_gammadashs, group_currents)

    _, pullback = jax.vjp(fwd, gammas, gammadashs, currents)
    return pullback(v)


# ── Grouped coil utilities ───────────────────────────────────────────


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


def _grouped_field(field_fn, points, coil_arrays):
    g0, gd0, c0 = coil_arrays[0]
    result = field_fn(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + field_fn(points, gammas, gammadashs, currents)
    return result


def grouped_biot_savart_B(points, coil_arrays):
    return _grouped_field(biot_savart_B, points, coil_arrays)


def grouped_biot_savart_A(points, coil_arrays):
    return _grouped_field(biot_savart_A, points, coil_arrays)
