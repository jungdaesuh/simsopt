"""
Pure JAX replacement for ``simsoptpp.integral_BdotN``.

Computes quadratic-flux-like surface integrals used in Stage-2 coil
optimization.  The three supported definitions are:

* ``"quadratic flux"``:
  ``J = 0.5 / (nphi·ntheta) · Σ (B·n̂ − B_T)² |n|``

* ``"normalized"``:
  ``J = 0.5 · Σ (B·n̂ − B_T)² |n|  /  Σ |B|² |n|``

* ``"local"``:
  ``J = 0.5 / (nphi·ntheta) · Σ (B·n̂ − B_T)² / |B|² · |n|``

All functions accept and return JAX arrays.
"""

import jax
import jax.numpy as jnp
from functools import partial

__all__ = ["integral_BdotN"]


@partial(jax.jit, static_argnames=("definition",))
def integral_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Compute the integral B·n objective.

    Args:
        Bcoil:  (nphi, ntheta, 3) coil magnetic field on the surface.
        target: (nphi, ntheta)    target normal field (can be zeros).
        normal: (nphi, ntheta, 3) unnormalized surface normal.
        definition: one of ``"quadratic flux"``, ``"normalized"``,
                    ``"local"``.  Treated as a compile-time constant
                    (static argument) for JIT tracing.

    Returns:
        J: scalar objective value.
    """
    nphi, ntheta, _ = Bcoil.shape

    norm_n = jnp.sqrt(jnp.sum(normal * normal, axis=-1))  # (nphi, ntheta)
    unit_n = normal / norm_n[..., None]  # (nphi, ntheta, 3)

    BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target  # (nphi, ntheta)

    if definition == "quadratic flux":
        return 0.5 * jnp.sum(BdotN * BdotN * norm_n) / (nphi * ntheta)

    elif definition == "normalized":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)  # (nphi, ntheta)
        numerator = jnp.sum(BdotN * BdotN * norm_n)
        denominator = jnp.sum(B2 * norm_n)
        return 0.5 * numerator / denominator

    elif definition == "local":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
        return 0.5 * jnp.sum(BdotN * BdotN / B2 * norm_n) / (nphi * ntheta)

    else:
        raise ValueError(f"Unknown definition: {definition!r}")
