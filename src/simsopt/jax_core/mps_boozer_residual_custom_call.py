"""Private MPS custom calls for SIMSOPT Boozer residual blocks."""

from __future__ import annotations

import jax
import numpy as np

SIMSOPT_MPS_BOOZER_RESIDUAL_VECTOR_TARGET = "mps.simsopt_boozer_residual_vector"
SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VECTOR_TARGET = (
    "mps.simsopt_boozer_residual_vector_weighted"
)
SIMSOPT_MPS_BOOZER_RESIDUAL_VJP_TARGET = "mps.simsopt_boozer_residual_vjp"
SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VJP_TARGET = (
    "mps.simsopt_boozer_residual_vjp_weighted"
)


def _validate_residual_inputs(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
) -> None:
    if G.shape != ():
        raise ValueError("G must be a scalar")
    if iota.shape != ():
        raise ValueError("iota must be a scalar")
    if len(B.shape) != 3 or B.shape[2] != 3:
        raise ValueError("B must have shape (nphi, ntheta, 3)")
    if B.shape[0] == 0 or B.shape[1] == 0:
        raise ValueError("B must include at least one surface grid point")
    if xphi.shape != B.shape:
        raise ValueError("xphi must have the same shape as B")
    if xtheta.shape != B.shape:
        raise ValueError("xtheta must have the same shape as B")
    if G.dtype != B.dtype or iota.dtype != B.dtype:
        raise ValueError("G, iota, and B must share a dtype")
    if xphi.dtype != B.dtype or xtheta.dtype != B.dtype:
        raise ValueError("B, xphi, and xtheta must share a dtype")
    if B.dtype != np.dtype(np.float32):
        raise ValueError("G, iota, B, xphi, and xtheta must be float32")


def _validate_residual_vjp_inputs(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
    residual_cotangent: jax.Array,
) -> None:
    _validate_residual_inputs(G, iota, B, xphi, xtheta)
    expected_shape = (int(np.prod(B.shape, dtype=np.int64)),)
    if residual_cotangent.shape != expected_shape:
        raise ValueError("residual_cotangent must have shape (nphi * ntheta * 3,)")
    if residual_cotangent.dtype != B.dtype:
        raise ValueError("residual_cotangent must have the same dtype as B")


def _residual_vector_target(*, weight_inv_modB: bool) -> str:
    if weight_inv_modB:
        return SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VECTOR_TARGET
    return SIMSOPT_MPS_BOOZER_RESIDUAL_VECTOR_TARGET


def _residual_vjp_target(*, weight_inv_modB: bool) -> str:
    if weight_inv_modB:
        return SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VJP_TARGET
    return SIMSOPT_MPS_BOOZER_RESIDUAL_VJP_TARGET


def simsopt_mps_boozer_residual_vector(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
    *,
    weight_inv_modB: bool = False,
) -> jax.Array:
    """Return the flattened Boozer residual vector through jax-mps."""
    _validate_residual_inputs(G, iota, B, xphi, xtheta)
    output_type = jax.ShapeDtypeStruct(
        (int(np.prod(B.shape, dtype=np.int64)),),
        B.dtype,
    )
    return jax.ffi.ffi_call(
        _residual_vector_target(weight_inv_modB=weight_inv_modB),
        output_type,
        vmap_method="broadcast_all",
        custom_call_api_version=4,
    )(G, iota, B, xphi, xtheta)


@jax.custom_vjp
def _simsopt_mps_boozer_residual_vector_with_unweighted_vjp(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
) -> jax.Array:
    return simsopt_mps_boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=False,
    )


def _simsopt_mps_boozer_residual_vector_unweighted_fwd(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    residual_vector = simsopt_mps_boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=False,
    )
    return residual_vector, (G, iota, B, xphi, xtheta)


def _simsopt_mps_boozer_residual_vector_unweighted_bwd(
    saved: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    residual_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    G, iota, B, xphi, xtheta = saved
    return simsopt_mps_boozer_residual_vjp(
        G,
        iota,
        B,
        xphi,
        xtheta,
        residual_cotangent,
        weight_inv_modB=False,
    )


_simsopt_mps_boozer_residual_vector_with_unweighted_vjp.defvjp(
    _simsopt_mps_boozer_residual_vector_unweighted_fwd,
    _simsopt_mps_boozer_residual_vector_unweighted_bwd,
)


@jax.custom_vjp
def _simsopt_mps_boozer_residual_vector_with_weighted_vjp(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
) -> jax.Array:
    return simsopt_mps_boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=True,
    )


def _simsopt_mps_boozer_residual_vector_weighted_fwd(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    residual_vector = simsopt_mps_boozer_residual_vector(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=True,
    )
    return residual_vector, (G, iota, B, xphi, xtheta)


def _simsopt_mps_boozer_residual_vector_weighted_bwd(
    saved: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    residual_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    G, iota, B, xphi, xtheta = saved
    return simsopt_mps_boozer_residual_vjp(
        G,
        iota,
        B,
        xphi,
        xtheta,
        residual_cotangent,
        weight_inv_modB=True,
    )


_simsopt_mps_boozer_residual_vector_with_weighted_vjp.defvjp(
    _simsopt_mps_boozer_residual_vector_weighted_fwd,
    _simsopt_mps_boozer_residual_vector_weighted_bwd,
)


def simsopt_mps_boozer_residual_vector_with_vjp(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
    *,
    weight_inv_modB: bool = False,
) -> jax.Array:
    """Return the residual vector with the paired jax-mps VJP as its AD rule."""
    _validate_residual_inputs(G, iota, B, xphi, xtheta)
    if weight_inv_modB:
        return _simsopt_mps_boozer_residual_vector_with_weighted_vjp(
            G,
            iota,
            B,
            xphi,
            xtheta,
        )
    return _simsopt_mps_boozer_residual_vector_with_unweighted_vjp(
        G,
        iota,
        B,
        xphi,
        xtheta,
    )


def simsopt_mps_boozer_residual_vjp(
    G: jax.Array,
    iota: jax.Array,
    B: jax.Array,
    xphi: jax.Array,
    xtheta: jax.Array,
    residual_cotangent: jax.Array,
    *,
    weight_inv_modB: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return Boozer residual-vector cotangents through jax-mps."""
    _validate_residual_vjp_inputs(G, iota, B, xphi, xtheta, residual_cotangent)
    output_type = (
        jax.ShapeDtypeStruct(G.shape, G.dtype),
        jax.ShapeDtypeStruct(iota.shape, iota.dtype),
        jax.ShapeDtypeStruct(B.shape, B.dtype),
        jax.ShapeDtypeStruct(xphi.shape, xphi.dtype),
        jax.ShapeDtypeStruct(xtheta.shape, xtheta.dtype),
    )
    return jax.ffi.ffi_call(
        _residual_vjp_target(weight_inv_modB=weight_inv_modB),
        output_type,
        vmap_method="broadcast_all",
        custom_call_api_version=4,
    )(G, iota, B, xphi, xtheta, residual_cotangent)
