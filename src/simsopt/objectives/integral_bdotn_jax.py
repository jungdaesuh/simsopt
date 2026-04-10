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

Zero-area quadrature points contribute zero. For ``"normalized"``,
nonpositive global ``Σ |B|² |n|`` is treated as invalid and returns
``inf``. For ``"local"``, any positive-area quadrature point with
``|B|² = 0`` is treated as invalid and also returns ``inf``.

All functions accept and return JAX arrays.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ..jax_core.reductions import (
    pairwise_sum_flat,
    scalar_square_sum,
    validate_reduction_mode,
)

__all__ = ["integral_BdotN", "residual_BdotN"]


@partial(jax.jit, static_argnames=("definition",))
def residual_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Return a least-squares residual vector for the selected flux definition."""
    nphi, ntheta, _ = Bcoil.shape

    norm_n = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    has_normal = norm_n > 0.0
    safe_norm_n = jnp.where(has_normal, norm_n, 1.0)
    unit_n = jnp.where(
        has_normal[..., None],
        normal / safe_norm_n[..., None],
        0.0,
    )
    BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target

    if definition == "quadratic flux":
        weight = jnp.where(has_normal, norm_n / (nphi * ntheta), 0.0)
        residual = jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0)
    elif definition == "normalized":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
        denominator = pairwise_sum_flat(B2 * norm_n)
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        point_weight = jnp.where(has_normal, norm_n / safe_denominator, 0.0)
        residual = jnp.where(
            denominator > 0.0,
            jnp.where(has_normal, BdotN * jnp.sqrt(point_weight), 0.0),
            jnp.full_like(BdotN, jnp.inf),
        )
    elif definition == "local":
        B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
        singular = has_normal & (B2 <= 0.0)
        safe_B2 = jnp.where(B2 > 0.0, B2, 1.0)
        weight = jnp.where(
            has_normal,
            norm_n / (safe_B2 * (nphi * ntheta)),
            0.0,
        )
        residual = jnp.where(
            singular,
            jnp.full_like(BdotN, jnp.inf),
            jnp.where(has_normal, BdotN * jnp.sqrt(weight), 0.0),
        )
    else:
        raise ValueError(f"Unknown definition: {definition!r}")

    return jnp.ravel(residual)


@partial(jax.jit, static_argnames=("definition", "reduction_mode"))
def integral_BdotN(
    Bcoil,
    target,
    normal,
    definition="quadratic flux",
    reduction_mode="default",
):
    """Compute the integral B·n objective.

    Args:
        Bcoil:  (nphi, ntheta, 3) coil magnetic field on the surface.
        target: (nphi, ntheta)    target normal field (can be zeros).
        normal: (nphi, ntheta, 3) unnormalized surface normal.
        definition: one of ``"quadratic flux"``, ``"normalized"``,
                    ``"local"``.  Treated as a compile-time constant
                    (static argument) for JIT tracing.
        reduction_mode: ``"default"`` keeps the kernel's validated hot-path
                    baseline, while ``"strict_oracle"`` enables the dedicated
                    compensated scalar objective contraction used for oracle
                    investigations.

    Returns:
        J: scalar objective value.
    """
    residual = residual_BdotN(
        Bcoil,
        target,
        normal,
        definition=definition,
    )
    validate_reduction_mode(reduction_mode)
    return 0.5 * scalar_square_sum(
        residual,
        reduction_mode=reduction_mode,
        default="vdot",
    )
