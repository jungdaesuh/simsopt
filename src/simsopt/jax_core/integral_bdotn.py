"""
Pure JAX replacement for ``simsoptpp.integral_BdotN``.

Computes quadratic-flux-like surface integrals used in Stage-2 coil
optimization.  The three supported definitions are:

* ``"quadratic flux"``:
  ``J = 0.5 / (nphi·ntheta) · Σ (B·n̂ − B_T)² |n|``

* ``"normalized"``:
  ``J = 0.5 · Σ (B·n̂ − B_T)² |n|  /  Σ |B|² |n|``

  This is algebraically equivalent to the C++ symmetric reduction but not a
  byte-identity contract: JAX stages the per-point least-squares residual and
  then contracts it, while C++ accumulates numerator and denominator through
  its own loop order. Use ``reduction_mode="strict_oracle"`` only for scalar
  contraction investigations; the default path preserves the AD-uniform
  residual formulation.

* ``"local"``:
  ``J = 0.5 / (nphi·ntheta) · Σ (B·n̂ − B_T)² / |B|² · |n|``

Zero-area quadrature points contribute zero. For ``"normalized"``,
nonpositive global ``Σ |B|² |n|`` is treated as invalid and returns
``inf``. For ``"local"``, any positive-area quadrature point with
``|B|² = 0`` is treated as invalid and also returns ``inf``.
Empty ``nphi`` or ``ntheta`` meshes return ``inf`` for ``"normalized"`` and
``nan`` for the other definitions, matching the undefined C++ reduction
contract.
An empty target array follows the C++ ``Btarget.size() == 0`` contract and is
interpreted as no target field.

All functions accept real JAX arrays. The pure-JAX path preserves real input
precision, including float32, while complex inputs are rejected because the
C++ oracle and Stage-2 flux objective are real-valued contracts.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P

from .reductions import (
    pairwise_sum_flat,
    scalar_square_sum,
    validate_reduction_mode,
)
from .sharding import (
    maybe_shard_surface_quadrature_inputs,
    surface_quadrature_sharding_config,
    surface_quadrature_sharding_summary,
)

__all__ = [
    "integral_BdotN",
    "integral_BdotN_surface_sharded",
    "integral_BdotN_sharding_summary",
    "residual_BdotN",
    "signed_BdotN_flux",
]

_VALID_DEFINITIONS = ("quadratic flux", "normalized", "local")


def _validate_bcoil_shape(Bcoil):
    if len(Bcoil.shape) != 3 or Bcoil.shape[2] != 3:
        raise ValueError(
            f"Bcoil must have shape (nphi, ntheta, 3); got Bcoil.shape={Bcoil.shape}."
        )


def _validate_normal_shape(Bcoil, normal):
    if normal.shape != Bcoil.shape:
        raise ValueError(
            "normal.shape must match Bcoil.shape; "
            f"got normal.shape={normal.shape}, Bcoil.shape={Bcoil.shape}."
        )


def _validate_target_shape(Bcoil, target):
    expected_shape = Bcoil.shape[:2]
    if target.shape != expected_shape:
        raise ValueError(
            "target.shape must match Bcoil.shape[:2]; "
            f"got target.shape={target.shape}, Bcoil.shape[:2]={expected_shape}."
        )


def _nan_safe_zero_grid(normal, dtype):
    """Return device-derived zeros without propagating NaNs from ``normal``."""
    finite_counts = jnp.sum(normal == normal, axis=-1)
    return (finite_counts - finite_counts).astype(dtype)


def _validate_real_dtype(name, value):
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be real-valued; got dtype {value.dtype}.")


def _validated_flux_target(Bcoil, target, normal):
    _validate_bcoil_shape(Bcoil)
    _validate_normal_shape(Bcoil, normal)
    _validate_real_dtype("Bcoil", Bcoil)
    _validate_real_dtype("target", target)
    _validate_real_dtype("normal", normal)
    if target.size == 0:
        return _nan_safe_zero_grid(normal, jnp.result_type(Bcoil, normal, 0.0))
    _validate_target_shape(Bcoil, target)
    return target


def _zero_scalar_like_flux(Bcoil, target, normal):
    zero_grid = _nan_safe_zero_grid(
        normal,
        jnp.result_type(Bcoil, target, normal, 0.0),
    )
    return pairwise_sum_flat(zero_grid)


def _validate_definition(definition):
    if definition not in _VALID_DEFINITIONS:
        raise ValueError(f"Unknown definition: {definition!r}")


def _masked_sqrt_weight_residual(mask, BdotN, weight):
    safe_weight = jnp.where(mask, weight, 1.0)
    safe_BdotN = jnp.where(mask, BdotN, 0.0)
    return jnp.where(mask, safe_BdotN * jnp.sqrt(safe_weight), 0.0)


@partial(jax.jit, static_argnames=("definition",))
def residual_BdotN(Bcoil, target, normal, definition="quadratic flux"):
    """Return a least-squares residual vector for the selected flux definition."""
    target = _validated_flux_target(Bcoil, target, normal)
    _validate_definition(definition)
    nphi, ntheta, _ = Bcoil.shape

    normal_norm2 = jnp.sum(normal * normal, axis=-1)
    has_normal = normal_norm2 > 0.0
    safe_norm_n = jnp.sqrt(jnp.where(has_normal, normal_norm2, 1.0))
    norm_n = jnp.where(has_normal, safe_norm_n, 0.0)
    unit_n = jnp.where(
        has_normal[..., None],
        normal / safe_norm_n[..., None],
        0.0,
    )
    safe_Bcoil = jnp.where(has_normal[..., None], Bcoil, 0.0)
    safe_target = jnp.where(has_normal, target, 0.0)
    BdotN = jnp.sum(safe_Bcoil * unit_n, axis=-1) - safe_target

    if definition == "quadratic flux":
        weight = jnp.where(has_normal, norm_n / (nphi * ntheta), 0.0)
        residual = _masked_sqrt_weight_residual(has_normal, BdotN, weight)
    elif definition == "normalized":
        B2 = jnp.sum(safe_Bcoil * safe_Bcoil, axis=-1)
        denominator = pairwise_sum_flat(B2 * norm_n)
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        point_weight = jnp.where(has_normal, norm_n / safe_denominator, 0.0)
        residual = jnp.where(
            denominator > 0.0,
            _masked_sqrt_weight_residual(has_normal, BdotN, point_weight),
            jnp.full_like(BdotN, jnp.inf),
        )
    elif definition == "local":
        B2 = jnp.sum(safe_Bcoil * safe_Bcoil, axis=-1)
        singular = has_normal & (B2 <= 0.0)
        safe_B2 = jnp.where(B2 > 0.0, B2, 1.0)
        weight = jnp.where(
            has_normal,
            norm_n / (safe_B2 * (nphi * ntheta)),
            0.0,
        )
        invalid_residual = jnp.reciprocal(jnp.where(singular, B2, jnp.ones_like(B2)))
        residual = jnp.where(
            singular,
            invalid_residual,
            _masked_sqrt_weight_residual(has_normal, BdotN, weight),
        )
    return jnp.ravel(residual)


@jax.jit
def signed_BdotN_flux(Bcoil, normal):
    """Return the raw signed average of B dot unnormalized surface normal."""
    _validate_bcoil_shape(Bcoil)
    _validate_normal_shape(Bcoil, normal)
    _validate_real_dtype("Bcoil", Bcoil)
    _validate_real_dtype("normal", normal)
    nphi, ntheta, _ = Bcoil.shape
    return pairwise_sum_flat(jnp.sum(Bcoil * normal, axis=-1)) / (nphi * ntheta)


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
    target = _validated_flux_target(Bcoil, target, normal)
    _validate_definition(definition)
    validate_reduction_mode(reduction_mode)
    if Bcoil.shape[0] == 0 or Bcoil.shape[1] == 0:
        zero = _zero_scalar_like_flux(Bcoil, target, normal)
        if definition == "normalized":
            return jnp.reciprocal(zero)
        return zero / zero
    residual = residual_BdotN(
        Bcoil,
        target,
        normal,
        definition=definition,
    )
    return 0.5 * scalar_square_sum(
        residual,
        reduction_mode=reduction_mode,
        default="vdot",
    )


@partial(jax.jit, static_argnames=("definition",))
def integral_BdotN_surface_sharded(Bcoil, target, normal, definition="quadratic flux"):
    """Compute ``integral_BdotN`` with explicit leading-surface-axis sharding."""
    target = _validated_flux_target(Bcoil, target, normal)
    _validate_definition(definition)
    config = surface_quadrature_sharding_config(Bcoil)
    if config is None:
        return integral_BdotN(Bcoil, target, normal, definition)
    nphi, ntheta, _ = Bcoil.shape

    Bcoil, target, normal = maybe_shard_surface_quadrature_inputs(
        Bcoil,
        target,
        normal,
        config=config,
    )

    @partial(
        jax.shard_map,
        mesh=config.mesh,
        in_specs=(
            P(config.axis_name, None, None),
            P(config.axis_name, None),
            P(config.axis_name, None, None),
        ),
        out_specs=P(),
        check_vma=True,
    )
    def integral_shard(Bcoil_block, target_block, normal_block):
        normal_norm2 = jnp.sum(normal_block * normal_block, axis=-1)
        has_normal = normal_norm2 > 0.0
        safe_norm_n = jnp.sqrt(jnp.where(has_normal, normal_norm2, 1.0))
        norm_n = jnp.where(has_normal, safe_norm_n, 0.0)
        unit_n = jnp.where(
            has_normal[..., None],
            normal_block / safe_norm_n[..., None],
            0.0,
        )
        safe_Bcoil = jnp.where(has_normal[..., None], Bcoil_block, 0.0)
        safe_target = jnp.where(has_normal, target_block, 0.0)
        BdotN = jnp.sum(safe_Bcoil * unit_n, axis=-1) - safe_target

        if definition == "quadratic flux":
            local_sum = jnp.sum(BdotN * BdotN * norm_n)
            total_sum = jax.lax.psum(local_sum, config.axis_name)
            return 0.5 * total_sum / (nphi * ntheta)
        if definition == "normalized":
            B2 = jnp.sum(safe_Bcoil * safe_Bcoil, axis=-1)
            numerator = jax.lax.psum(
                jnp.sum(BdotN * BdotN * norm_n),
                config.axis_name,
            )
            denominator = jax.lax.psum(
                jnp.sum(B2 * norm_n),
                config.axis_name,
            )
            return jnp.where(
                denominator > 0.0,
                0.5 * numerator / denominator,
                jnp.asarray(jnp.inf, dtype=Bcoil_block.dtype),
            )

        B2 = jnp.sum(safe_Bcoil * safe_Bcoil, axis=-1)
        singular = has_normal & (B2 <= 0.0)
        safe_B2 = jnp.where(B2 > 0.0, B2, 1.0)
        local_sum = jnp.sum(jnp.where(has_normal, BdotN * BdotN * norm_n / safe_B2, 0.0))
        total_sum = jax.lax.psum(local_sum, config.axis_name)
        invalid_count = jax.lax.psum(jnp.sum(singular.astype(jnp.int32)), config.axis_name)
        return jnp.where(
            invalid_count > 0,
            jnp.asarray(jnp.inf, dtype=Bcoil_block.dtype),
            0.5 * total_sum / (nphi * ntheta),
        )

    return integral_shard(Bcoil, target, normal)


def integral_BdotN_sharding_summary(
    Bcoil,
    target,
    normal,
    definition="quadratic flux",
) -> dict[str, object]:
    """Return the surface-quadrature sharding summary for one integral call."""
    del target, normal, definition
    config = surface_quadrature_sharding_config(Bcoil)
    if config is None:
        return surface_quadrature_sharding_summary(Bcoil, config=config)
    (sharded_Bcoil,) = maybe_shard_surface_quadrature_inputs(Bcoil, config=config)
    return surface_quadrature_sharding_summary(sharded_Bcoil, config=config)
