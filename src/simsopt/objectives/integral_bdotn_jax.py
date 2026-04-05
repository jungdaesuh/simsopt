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

__all__ = ["integral_BdotN", "residual_BdotN"]


@partial(jax.jit, static_argnames=("definition",))
def residual_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Return a least-squares residual vector for the selected flux definition."""
    nphi, ntheta, _ = Bcoil.shape

    norm_n = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    unit_n = normal / norm_n[..., None]
    BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target

    if definition == "quadratic flux":
        weight = norm_n / (nphi * ntheta)
        residual = BdotN * jnp.sqrt(weight)
    elif definition == "normalized":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
        denominator = jnp.sum(B2 * norm_n)
        residual = BdotN * jnp.sqrt(norm_n / denominator)
    elif definition == "local":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
        safe_B2 = jnp.where(B2 > 0.0, B2, 1.0)
        inv_B2 = jnp.where(B2 > 0.0, 1.0 / safe_B2, 0.0)
        residual = BdotN * jnp.sqrt(inv_B2 * norm_n / (nphi * ntheta))
    else:
        raise ValueError(f"Unknown definition: {definition!r}")

    return jnp.ravel(residual)


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
    residual = residual_BdotN(
        Bcoil,
        target,
        normal,
        definition=definition,
    )
    return 0.5 * jnp.vdot(residual, residual).real
