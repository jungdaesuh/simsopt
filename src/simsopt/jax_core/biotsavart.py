"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
replace the C++ ``simsoptpp.BiotSavart`` kernel for GPU execution.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

from enum import Enum
from functools import lru_cache

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from ..backend import (
    get_field_kernel_tuning,
    get_point_chunk_size,
)
from ..backend.runtime import register_backend_cache_clear
from ._math_utils import (
    explicit_inv as _explicit_inv,
    explicit_rsqrt as _explicit_rsqrt,
    eye as _eye,
    scalar_like as _scalar_like,
    zeros as _zeros,
)

__all__ = [
    "biot_savart_B",
    "biot_savart_B_vjp",
    "biot_savart_dB_by_dX",
    "biot_savart_d2B_by_dXdX",
    "biot_savart_B_and_dB",
    "biot_savart_A",
    "biot_savart_dA_by_dX",
    "biot_savart_d2A_by_dXdX",
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
    indices = _as_int32_scalar(start) + _index_range(chunk_size)
    return jnp.take(array, indices, axis=0)


def _slice_quadrature_block(array, start: int, block_size: int):
    indices = _as_int32_scalar(start) + _index_range(block_size)
    return jnp.take(array, indices, axis=1)


def _slice_point_chunk(points: object, start: int, chunk_size: int):
    indices = _as_int32_scalar(start) + _index_range(chunk_size)
    return jnp.take(points, indices, axis=0)


def _slice_prefix(array, size: int):
    return jnp.take(array, _index_range(size), axis=0)


def _float64_scalar(value):
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _as_int32_scalar(value):
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.int32)
    return jax.device_put(np.asarray(value, dtype=np.int32))


def _index_range(size: int):
    return jax.device_put(np.arange(size, dtype=np.int32))


def _zero_scalar(dtype):
    return jax.device_put(np.array(0, dtype=np.dtype(dtype)))


def _vector_component(array, component_index: int):
    return array[..., component_index]


def _cross_product(left, right):
    left_x = _vector_component(left, 0)
    left_y = _vector_component(left, 1)
    left_z = _vector_component(left, 2)
    right_x = _vector_component(right, 0)
    right_y = _vector_component(right, 1)
    right_z = _vector_component(right, 2)
    return jnp.stack(
        (
            left_y * right_z - left_z * right_y,
            left_z * right_x - left_x * right_z,
            left_x * right_y - left_y * right_x,
        ),
        axis=-1,
    )


def _pad_axis0(array, padded_size: int):
    pad_rows = padded_size - array.shape[0]
    if pad_rows <= 0:
        return array
    padding_config = [(0, pad_rows, 0)] + [(0, 0, 0)] * (array.ndim - 1)
    return lax.pad(array, _zero_scalar(array.dtype), padding_config)


def _pad_axis1(array, padded_size: int):
    pad_cols = padded_size - array.shape[1]
    if pad_cols <= 0:
        return array
    padding_config = [(0, 0, 0), (0, pad_cols, 0)] + [(0, 0, 0)] * (array.ndim - 2)
    return lax.pad(array, _zero_scalar(array.dtype), padding_config)


def _next_power_of_two(size: int) -> int:
    if size <= 1:
        return 1
    return 1 << (size - 1).bit_length()


def _pairwise_sum_axis(array, *, axis: int):
    """Reduce ``array`` along ``axis`` using a fixed binary addition tree."""
    axis_index = axis if axis >= 0 else array.ndim + axis
    axis_size = array.shape[axis_index]
    if axis_size == 0:
        return jnp.sum(array, axis=axis_index)

    reduced = jnp.moveaxis(array, axis_index, 0)
    reduced = _pad_axis0(reduced, _next_power_of_two(axis_size))
    while reduced.shape[0] > 1:
        pair_shape = (reduced.shape[0] // 2, 2) + tuple(reduced.shape[1:])
        paired = jnp.reshape(reduced, pair_shape)
        reduced = paired[:, 0, ...] + paired[:, 1, ...]
    return jnp.squeeze(reduced, axis=0)


# ── Tree utilities ────────────────────────────────────────────────────


def _tree_dynamic_update(prefix_tree, chunk_tree, start_index: int):
    return jax.tree_util.tree_map(
        lambda acc, update: lax.dynamic_update_slice(
            acc,
            update,
            (_as_int32_scalar(start_index),) + (_as_int32_scalar(0),) * (acc.ndim - 1),
        ),
        prefix_tree,
        chunk_tree,
    )


def _tree_trim(prefix_tree, size: int):
    return jax.tree_util.tree_map(lambda leaf: _slice_prefix(leaf, size), prefix_tree)


def _tree_concatenate(left, right):
    return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=0), left, right
    )


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree_util.tree_map(
        lambda leaf: _zeros(
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

    padded_coil_count = chunk_count * chunk_size
    padded_gammas = _pad_axis0(gammas, padded_coil_count)
    padded_gammadashs = _pad_axis0(gammadashs, padded_coil_count)
    padded_currents = _pad_axis0(currents, padded_coil_count)

    # Keep the outer coil accumulation serial until parity data implicates it;
    # the quadrature-axis sum dominates the known reduction-order drift, while
    # tree-reducing chunk outputs would add another staged hot-path combine.
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
        values = integrand(x, gammas, gammadashs)
        return _pairwise_sum_axis(values, axis=1) * _scalar_like(
            values,
            1.0 / values.shape[1],
        )

    block_count = (quadrature_count + block_size - 1) // block_size
    if block_count == 2:
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

    padded_quadrature_count = block_count * block_size
    padded_gammas = _pad_axis1(gammas, padded_quadrature_count)
    padded_gammadashs = _pad_axis1(gammadashs, padded_quadrature_count)
    zero = _zeros((gammas.shape[0], 3), dtype=jnp.float64)

    def body(block_index: int, acc):
        start = block_index * block_size
        block_gammas = _slice_quadrature_block(padded_gammas, start, block_size)
        block_gammadashs = _slice_quadrature_block(
            padded_gammadashs,
            start,
            block_size,
        )
        block_integrand = integrand(x, block_gammas, block_gammadashs)
        return acc + _pairwise_sum_axis(block_integrand, axis=1)

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
    padded_point_count = chunk_count * chunk_size
    padded_points = _pad_axis0(points, padded_point_count)
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
    # Exact point-on-coil evaluation is outside the physical domain; use a tiny
    # branchless floor so traced audit lanes do not rely on select/where.
    safe_r2 = r2 + _float64_scalar(np.finfo(np.float64).tiny)
    r_inv = _explicit_rsqrt(safe_r2)
    r_inv3 = r_inv * _explicit_inv(safe_r2)
    cross = _cross_product(diff, gammadashs)
    return cross * r_inv3[..., None]


def _biot_savart_A_integrand(x, gammas, gammadashs):
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = r2 + _float64_scalar(np.finfo(np.float64).tiny)
    r_inv = _explicit_rsqrt(safe_r2)
    return gammadashs * r_inv[..., None]


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
    return _float64_scalar(_MU0_OVER_4PI) * jnp.einsum("c,cj->j", currents, integral)


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

        def per_point(x, gammas, gammadashs, currents):
            return jnp.swapaxes(
                jax.jacfwd(one_point, argnums=0)(x, gammas, gammadashs, currents),
                -1,
                -2,
            )

    elif diff_mode is _DiffMode.HESSIAN:

        def per_point(x, gammas, gammadashs, currents):
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

    elif diff_mode is _DiffMode.VALUE_AND_JACOBIAN:

        def per_point(x, gammas, gammadashs, currents):
            f = lambda xx: one_point(xx, gammas, gammadashs, currents)
            primals, tangents_fn = jax.linearize(f, x)
            return primals, jax.vmap(tangents_fn)(_eye(3, dtype=jnp.float64))

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


@lru_cache(maxsize=16)
def _make_B_vjp_kernel(coil_cs, quad_bs, point_cs):
    """Build the tuning-keyed compiled VJP kernel for ``biot_savart_B``.

    ``biot_savart_B_vjp`` differentiates through the forward field kernel, so it
    must be rebuilt when the backend tuning changes in the same process.  Keying
    this factory on the same tuning tuple as the forward kernels keeps the
    backend/performance contract consistent across both paths.
    """
    forward_kernel = _make_kernel(
        _Integrand.B,
        _DiffMode.VALUE,
        coil_cs,
        quad_bs,
        point_cs,
    )

    @jax.jit
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

    return kernel


def _get_B_vjp_kernel():
    """Read tuning config and return the cached JIT-compiled VJP kernel."""
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    return _make_B_vjp_kernel(coil_cs, quad_bs, point_cs)


def invalidate_kernel_cache() -> None:
    """Drop all cached JIT-compiled Biot-Savart kernels and tuning config.

    Call after overriding ``_read_tuning_config`` (e.g. via ``monkeypatch``)
    to ensure the next ``biot_savart_*`` call rebuilds with the new config.
    """
    _make_kernel.cache_clear()
    _make_B_vjp_kernel.cache_clear()


register_backend_cache_clear(invalidate_kernel_cache)


# ── Public API ────────────────────────────────────────────────────────


def biot_savart_B(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.B, _DiffMode.VALUE)(
        points, gammas, gammadashs, currents
    )


def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.B, _DiffMode.JACOBIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_d2B_by_dXdX(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.B, _DiffMode.HESSIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.B, _DiffMode.VALUE_AND_JACOBIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_A(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.A, _DiffMode.VALUE)(
        points, gammas, gammadashs, currents
    )


def biot_savart_dA_by_dX(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.A, _DiffMode.JACOBIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_d2A_by_dXdX(points, gammas, gammadashs, currents):
    return _get_kernel(_Integrand.A, _DiffMode.HESSIAN)(
        points, gammas, gammadashs, currents
    )


def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """VJP of ``biot_savart_B`` w.r.t. coil data.

    Uses a dedicated tuning-keyed kernel factory so backend mode and chunking
    changes rebuild the compiled closure in the same process, matching the
    cache invalidation behavior of the forward ``biot_savart_B`` kernels.
    """
    return _get_B_vjp_kernel()(points, v, gammas, gammadashs, currents)


# ── Grouped coil utilities ───────────────────────────────────────────


def _axis0_entries(array: object) -> tuple[jax.Array, ...]:
    array_jax = jnp.asarray(array)
    if array_jax.ndim == 0:
        return (array_jax,)
    length = int(array_jax.shape[0])
    if length == 0:
        return ()
    return tuple(
        jnp.squeeze(chunk, axis=0)
        for chunk in jnp.split(array_jax, length, axis=0)
    )


def _coil_entry_sequence(values: object) -> tuple[object, ...]:
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return _axis0_entries(values)


def group_coil_data(gammas_list, gammadashs_list, currents_list):
    gamma_entries = _coil_entry_sequence(gammas_list)
    gammadash_entries = _coil_entry_sequence(gammadashs_list)
    current_entries = _coil_entry_sequence(currents_list)
    by_nquad = {}
    for i, gamma in enumerate(gamma_entries):
        by_nquad.setdefault(gamma.shape[0], []).append(i)

    groups = []
    for indices in by_nquad.values():
        groups.append(
            (
                jnp.stack(
                    [jnp.asarray(gamma_entries[i], dtype=jnp.float64) for i in indices]
                ),
                jnp.stack(
                    [
                        jnp.asarray(gammadash_entries[i], dtype=jnp.float64)
                        for i in indices
                    ]
                ),
                jnp.stack(
                    [jnp.asarray(current_entries[i], dtype=jnp.float64) for i in indices]
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
