"""Analytic Boozer least-squares derivatives in intrinsic surface coordinates."""

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "boozer_residual_analytic_value_grad",
    "boozer_residual_analytic_value_grad_hessian",
]


def _unweighted_local_residual(local_variables):
    B = local_variables[:3]
    tang = local_variables[3:6]
    G = local_variables[6]
    return G * B - jnp.sum(B * B) * tang


def _weighted_local_residual(local_variables):
    B = local_variables[:3]
    return _unweighted_local_residual(local_variables) / jnp.sqrt(jnp.sum(B * B))


_UNWEIGHTED_LOCAL_JACOBIAN = jax.jacfwd(_unweighted_local_residual)
_UNWEIGHTED_LOCAL_HESSIAN = jax.jacfwd(_UNWEIGHTED_LOCAL_JACOBIAN)
_WEIGHTED_LOCAL_JACOBIAN = jax.jacfwd(_weighted_local_residual)
_WEIGHTED_LOCAL_HESSIAN = jax.jacfwd(_WEIGHTED_LOCAL_JACOBIAN)


def _local_residual_derivatives(B, tang, G, *, weight_inv_modB, order):
    """Differentiate only the fixed-size pointwise 7-to-3 residual map."""
    local_variables = jnp.concatenate(
        (B, tang, jnp.broadcast_to(jnp.asarray(G), (B.shape[0], 1))),
        axis=-1,
    )
    if weight_inv_modB:
        residual_fn = _weighted_local_residual
        jacobian_fn = _WEIGHTED_LOCAL_JACOBIAN
        hessian_fn = _WEIGHTED_LOCAL_HESSIAN
    else:
        residual_fn = _unweighted_local_residual
        jacobian_fn = _UNWEIGHTED_LOCAL_JACOBIAN
        hessian_fn = _UNWEIGHTED_LOCAL_HESSIAN
    residual = jax.vmap(residual_fn)(local_variables)
    local_jacobian = jax.vmap(jacobian_fn)(local_variables)
    value = 0.5 * jnp.sum(residual * residual, axis=-1)
    gradient = jnp.einsum("pk,pki->pi", residual, local_jacobian)
    if order == 1:
        return value, gradient, None

    residual_hessian = jax.vmap(hessian_fn)(local_variables)
    hessian = (
        jnp.einsum("pki,pkj->pij", local_jacobian, local_jacobian)
        + jnp.einsum("pk,pkij->pij", residual, residual_hessian)
    )
    return value, gradient, hessian


def _tile_linearization(
    dB_dX,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    iota,
    *,
    optimize_G,
):
    """Return the pointwise Jacobian of (B, tang, G) in decision-vector order."""
    dB_ds = jnp.einsum("pdc,pds->pcs", dB_dX, dx_ds)
    dtang_ds = dxphi_ds + iota * dxtheta_ds
    surface_columns = jnp.concatenate((dB_ds, dtang_ds), axis=1)
    zero_scalar_column = jnp.zeros(
        (surface_columns.shape[0], 1, surface_columns.shape[-1]),
        dtype=surface_columns.dtype,
    )
    surface_columns = jnp.concatenate((surface_columns, zero_scalar_column), axis=1)
    iota_column = jnp.concatenate(
        (jnp.zeros_like(xtheta), xtheta, jnp.zeros_like(xtheta[:, :1])),
        axis=-1,
    )[..., None]
    if not optimize_G:
        return jnp.concatenate((surface_columns, iota_column), axis=-1), dB_ds

    G_column = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=dB_dX.dtype),
        (dB_dX.shape[0], 7),
    )[..., None]
    return jnp.concatenate((surface_columns, iota_column, G_column), axis=-1), dB_ds


def _flatten_inputs(B, dB_dX, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds):
    npoints = B.shape[0] * B.shape[1]
    nsurface = dx_ds.shape[-1]
    return (
        jnp.reshape(B, (npoints, 3)),
        jnp.reshape(dB_dX, (npoints, 3, 3)),
        jnp.reshape(xphi, (npoints, 3)),
        jnp.reshape(xtheta, (npoints, 3)),
        jnp.reshape(dx_ds, (npoints, 3, nsurface)),
        jnp.reshape(dxphi_ds, (npoints, 3, nsurface)),
        jnp.reshape(dxtheta_ds, (npoints, 3, nsurface)),
    )


def _dynamic_tile(array, tile_index, tile_size):
    return lax.dynamic_slice_in_dim(array, tile_index * tile_size, tile_size, axis=0)


def _value_grad_tile(
    G,
    iota,
    B,
    dB_dX,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    *,
    optimize_G,
    weight_inv_modB,
):
    tang = xphi + iota * xtheta
    tile_value, local_gradient, _local_hessian = _local_residual_derivatives(
        B,
        tang,
        G,
        weight_inv_modB=weight_inv_modB,
        order=1,
    )
    linearization, _dB_ds = _tile_linearization(
        dB_dX,
        xtheta,
        dx_ds,
        dxphi_ds,
        dxtheta_ds,
        iota,
        optimize_G=optimize_G,
    )
    gradient = jnp.reshape(local_gradient, (-1,)) @ jnp.reshape(
        linearization,
        (-1, linearization.shape[-1]),
    )
    return jnp.sum(tile_value), gradient


def _value_grad_hessian_tile(
    G,
    iota,
    B,
    dB_dX,
    d2B_dXdX,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    *,
    optimize_G,
    weight_inv_modB,
):
    tang = xphi + iota * xtheta
    tile_value, local_gradient, local_hessian = _local_residual_derivatives(
        B,
        tang,
        G,
        weight_inv_modB=weight_inv_modB,
        order=2,
    )
    linearization, _dB_ds = _tile_linearization(
        dB_dX,
        xtheta,
        dx_ds,
        dxphi_ds,
        dxtheta_ds,
        iota,
        optimize_G=optimize_G,
    )
    linearization_flat = jnp.reshape(
        linearization,
        (-1, linearization.shape[-1]),
    )
    weighted_linearization = jnp.einsum(
        "pij,pjv->piv",
        local_hessian,
        linearization,
    )
    hessian = linearization_flat.T @ jnp.reshape(
        weighted_linearization,
        (-1, linearization.shape[-1]),
    )

    field_position_hessian = jnp.einsum(
        "pk,pabk->pab",
        local_gradient[:, :3],
        d2B_dXdX,
    )
    weighted_surface_directions = jnp.einsum(
        "pab,pbs->pas",
        field_position_hessian,
        dx_ds,
    )
    surface_curvature = jnp.reshape(dx_ds, (-1, dx_ds.shape[-1])).T @ jnp.reshape(
        weighted_surface_directions,
        (-1, dx_ds.shape[-1]),
    )
    nsurface = dx_ds.shape[-1]
    hessian = hessian.at[:nsurface, :nsurface].add(surface_curvature)
    surface_iota_curvature = jnp.reshape(local_gradient[:, 3:6], (-1,)) @ jnp.reshape(
        dxtheta_ds,
        (-1, nsurface),
    )
    hessian = hessian.at[:nsurface, nsurface].add(surface_iota_curvature)
    hessian = hessian.at[nsurface, :nsurface].add(surface_iota_curvature)
    gradient = jnp.reshape(local_gradient, (-1,)) @ linearization_flat
    return jnp.sum(tile_value), gradient, hessian


def boozer_residual_analytic_value_grad(
    G,
    iota,
    B,
    dB_dX,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    *,
    optimize_G=True,
    weight_inv_modB=False,
    quadrature_tile_size=64,
):
    """Return normalized Boozer LS value and gradient for [surface, iota, (G)]."""
    flat = _flatten_inputs(
        B,
        dB_dX,
        xphi,
        xtheta,
        dx_ds,
        dxphi_ds,
        dxtheta_ds,
    )
    B_flat = flat[0]
    dx_ds_flat = flat[4]
    npoints = B_flat.shape[0]
    nvariables = dx_ds_flat.shape[-1] + 1 + int(optimize_G)
    value = jnp.zeros((), dtype=B_flat.dtype)
    gradient = jnp.zeros((nvariables,), dtype=B_flat.dtype)

    tile_size = min(quadrature_tile_size, npoints)
    full_tile_count = npoints // tile_size

    def accumulate_full_tile(tile_index, accumulator):
        tile_inputs = tuple(
            _dynamic_tile(array, tile_index, tile_size)
            for array in flat
        )
        tile_value, tile_gradient = _value_grad_tile(
            G,
            iota,
            *tile_inputs,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
        return accumulator[0] + tile_value, accumulator[1] + tile_gradient

    value, gradient = lax.fori_loop(
        0,
        full_tile_count,
        accumulate_full_tile,
        (value, gradient),
    )
    tail_start = full_tile_count * tile_size
    if tail_start < npoints:
        tail_inputs = tuple(array[tail_start:] for array in flat)
        tail_value, tail_gradient = _value_grad_tile(
            G,
            iota,
            *tail_inputs,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
        value = value + tail_value
        gradient = gradient + tail_gradient

    normalization = 3 * B.shape[0] * B.shape[1]
    return value / normalization, gradient / normalization


def boozer_residual_analytic_value_grad_hessian(
    G,
    iota,
    B,
    dB_dX,
    d2B_dXdX,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    *,
    optimize_G=True,
    weight_inv_modB=False,
    quadrature_tile_size=64,
):
    """Return normalized Boozer LS value, gradient, and true analytic Hessian."""
    flat = _flatten_inputs(
        B,
        dB_dX,
        xphi,
        xtheta,
        dx_ds,
        dxphi_ds,
        dxtheta_ds,
    )
    B_flat, dB_dX_flat, xphi_flat, xtheta_flat = flat[:4]
    dx_ds_flat, dxphi_ds_flat, dxtheta_ds_flat = flat[4:]
    npoints = B_flat.shape[0]
    nsurface = dx_ds_flat.shape[-1]
    nvariables = nsurface + 1 + int(optimize_G)
    d2B_flat = jnp.reshape(d2B_dXdX, (npoints, 3, 3, 3))
    value = jnp.zeros((), dtype=B_flat.dtype)
    gradient = jnp.zeros((nvariables,), dtype=B_flat.dtype)
    hessian = jnp.zeros((nvariables, nvariables), dtype=B_flat.dtype)

    tile_size = min(quadrature_tile_size, npoints)
    full_tile_count = npoints // tile_size
    hessian_flat = (
        B_flat,
        dB_dX_flat,
        d2B_flat,
        xphi_flat,
        xtheta_flat,
        dx_ds_flat,
        dxphi_ds_flat,
        dxtheta_ds_flat,
    )

    def accumulate_full_tile(tile_index, accumulator):
        tile_inputs = tuple(
            _dynamic_tile(array, tile_index, tile_size)
            for array in hessian_flat
        )
        tile_value, tile_gradient, tile_hessian = _value_grad_hessian_tile(
            G,
            iota,
            *tile_inputs,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
        return (
            accumulator[0] + tile_value,
            accumulator[1] + tile_gradient,
            accumulator[2] + tile_hessian,
        )

    value, gradient, hessian = lax.fori_loop(
        0,
        full_tile_count,
        accumulate_full_tile,
        (value, gradient, hessian),
    )
    tail_start = full_tile_count * tile_size
    if tail_start < npoints:
        tail_inputs = tuple(array[tail_start:] for array in hessian_flat)
        tail_value, tail_gradient, tail_hessian = _value_grad_hessian_tile(
            G,
            iota,
            *tail_inputs,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )
        value = value + tail_value
        gradient = gradient + tail_gradient
        hessian = hessian + tail_hessian

    normalization = 3 * B.shape[0] * B.shape[1]
    return value / normalization, gradient / normalization, hessian / normalization
