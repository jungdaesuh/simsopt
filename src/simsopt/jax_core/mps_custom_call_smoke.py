"""Private smoke op for proving SIMSOPT custom-call routing on jax-mps."""

from __future__ import annotations

import jax

SIMSOPT_MPS_CUSTOM_CALL_SMOKE_TARGET = "mps.simsopt_custom_call_smoke"


def simsopt_mps_custom_call_smoke(x: jax.Array) -> jax.Array:
    """Return `x` through the experimental jax-mps SIMSOPT custom-call target."""
    return jax.ffi.ffi_call(
        SIMSOPT_MPS_CUSTOM_CALL_SMOKE_TARGET,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
        custom_call_api_version=4,
    )(x)
