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

# μ₀ / (4π) in SI units  [T·m/A]
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
    """Dense Biot-Savart B at a single evaluation point."""
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
    """Biot-Savart B at a single evaluation point."""
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
    """Compute the Biot-Savart magnetic field at many evaluation points.

    .. math::

        \\mathbf B(\\mathbf x) = \\frac{\\mu_0}{4\\pi}
        \\sum_k I_k \\int_0^1
        \\frac{(\\Gamma_k - \\mathbf x) \\times \\Gamma_k'}{
               \\|\\Gamma_k - \\mathbf x\\|^3}\\,d\\varphi

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        B: (npoints, 3) magnetic field [T].
    """
    return _biot_savart_B_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )


@jax.jit
def biot_savart_B_vjp(points, v, gammas, gammadashs, currents):
    """Vector-Jacobian product of ``biot_savart_B`` w.r.t. grouped coil inputs."""

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
    """Compute the spatial Jacobian dB/dX at many evaluation points.

    Uses forward-mode autodiff on the single-point kernel (3→3, so
    ``jacfwd`` requires exactly 3 JVP evaluations).

    Follows the SIMSOPT convention from ``fields.rst``:
    ``dB_dX[p, j, l] = ∂_j B_l(x_p)``, i.e. axis 1 is the derivative
    direction, axis 2 is the B component.

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        dB_dX: (npoints, 3, 3) where ``dB_dX[p, j, l] = ∂_j B_l``.
    """
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
    """Compute B and dB/dX together (shares JIT compilation overhead).

    Returns:
        (B, dB_dX) with shapes (npoints, 3) and (npoints, 3, 3),
        where ``dB_dX[p, j, l] = ∂_j B_l`` (SIMSOPT convention).
    """
    B, dB_dX = _biot_savart_B_and_dB_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )
    return B, dB_dX


# ---------------------------------------------------------------------------
# Vector potential  A(x) = (μ₀/4π) Σ_k I_k ∫ Γ'_k / |Γ_k − x| dφ
# ---------------------------------------------------------------------------


def _biot_savart_A_one_point_dense(x, gammas, gammadashs, currents):
    """Dense Biot-Savart vector potential A at a single evaluation point."""
    diff = gammas - x
    r2 = jnp.sum(diff * diff, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv = jnp.where(r2 > 0, safe_r2 ** (-0.5), 0.0)
    integrand = gammadashs * r_inv[..., None]
    integral = jnp.mean(integrand, axis=1)
    return _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)


def _biot_savart_A_one_point(x, gammas, gammadashs, currents, *, chunk_size: int):
    """Biot-Savart vector potential A at a single evaluation point."""
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
    """Compute the Biot-Savart vector potential at many evaluation points.

    .. math::

        \\mathbf A(\\mathbf x) = \\frac{\\mu_0}{4\\pi}
        \\sum_k I_k \\int_0^1
        \\frac{\\Gamma_k'}{\\|\\Gamma_k - \\mathbf x\\|}\\,d\\varphi

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        A: (npoints, 3) vector potential [T·m].
    """
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
    """Compute the spatial Jacobian dA/dX at many evaluation points.

    Follows SIMSOPT convention: ``dA_dX[p, j, l] = ∂_j A_l(x_p)``.

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        dA_dX: (npoints, 3, 3) where ``dA_dX[p, j, l] = ∂_j A_l``.
    """
    return _biot_savart_dA_by_dX_impl(
        points,
        gammas,
        gammadashs,
        currents,
        chunk_size=_coil_chunk_size(),
    )


# ---------------------------------------------------------------------------
# Grouped coil utilities for mixed-quadrature support
# ---------------------------------------------------------------------------


def group_coil_data(gammas_list, gammadashs_list, currents_list):
    """Group per-coil geometry arrays by quadrature point count.

    The CPU Biot-Savart kernel evaluates each coil individually with its
    own ``num_quad_points``.  The JAX kernels need rectangular batches.
    This function groups coils that share the same quadrature count so
    they can be stacked into ``(n_coils_in_group, nquad, 3)`` arrays,
    then each group is evaluated separately and the results summed.

    Args:
        gammas_list: list of ``(nquad_i, 3)`` NumPy arrays.
        gammadashs_list: list of ``(nquad_i, 3)`` NumPy arrays.
        currents_list: list of float scalars.

    Returns:
        list of ``(gammas, gammadashs, currents, indices)`` tuples.
        ``indices`` maps each position in the group back to the original
        coil index in the input lists.
    """
    by_nquad = {}
    for i, g in enumerate(gammas_list):
        by_nquad.setdefault(g.shape[0], []).append(i)

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
    """Sum ``biot_savart_B`` over coil groups with different quadrature.

    Args:
        points: ``(npoints, 3)`` evaluation points.
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples,
            one per quadrature-count group.  Typically the first three
            elements of each tuple returned by :func:`group_coil_data`.

    Returns:
        ``(npoints, 3)`` total magnetic field.
    """
    g0, gd0, c0 = coil_arrays[0]
    result = biot_savart_B(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + biot_savart_B(points, gammas, gammadashs, currents)
    return result


def grouped_biot_savart_A(points, coil_arrays):
    """Sum ``biot_savart_A`` over coil groups with different quadrature.

    Same interface as :func:`grouped_biot_savart_B` but for the vector
    potential.
    """
    g0, gd0, c0 = coil_arrays[0]
    result = biot_savart_A(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + biot_savart_A(points, gammas, gammadashs, currents)
    return result
