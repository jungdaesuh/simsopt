"""Memory-bounded pure-JAX stochastic Stage-II flux reductions."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral_from_B
from simsopt_jax.core.specs import FixedSurfaceFluxSpec


@dataclass(frozen=True, slots=True)
class StochasticCoilPerturbations:
    """Fixed sampled perturbations for every final coil and quadrature point."""

    gamma: jax.Array
    gammadash: jax.Array

    def __post_init__(self) -> None:
        if self.gamma.ndim != 4 or self.gamma.shape[-1] != 3:
            raise ValueError("gamma must have shape (samples, coils, points, 3)")
        if self.gammadash.shape != self.gamma.shape:
            raise ValueError("gammadash must have the same shape as gamma")
        if self.gamma.shape[0] < 1:
            raise ValueError("at least one stochastic sample is required")


jax.tree_util.register_dataclass(
    StochasticCoilPerturbations,
    data_fields=("gamma", "gammadash"),
    meta_fields=(),
)


def stochastic_flux_mean_from_geometry(
    gamma: jax.Array,
    gammadash: jax.Array,
    currents: jax.Array,
    flux_spec: FixedSurfaceFluxSpec,
    perturbations: StochasticCoilPerturbations,
) -> jax.Array:
    """Evaluate a sample mean without retaining every sample field in memory."""
    gamma = jnp.asarray(gamma)
    gammadash = jnp.asarray(gammadash, dtype=gamma.dtype)
    currents = jnp.asarray(currents, dtype=gamma.dtype)
    zero = jnp.sum(gamma[:0])

    def accumulate(
        total: jax.Array,
        sample: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        gamma_perturbation, gammadash_perturbation = sample
        field = biot_savart_B(
            flux_spec.points,
            gamma + gamma_perturbation,
            gammadash + gammadash_perturbation,
            currents,
        )
        return total + fixed_surface_flux_integral_from_B(field, flux_spec), None

    total, _ = jax.lax.scan(
        jax.checkpoint(accumulate),
        zero,
        (perturbations.gamma, perturbations.gammadash),
    )
    sample_count = jnp.asarray(perturbations.gamma.shape[0], dtype=total.dtype)
    return total / sample_count


__all__ = (
    "StochasticCoilPerturbations",
    "stochastic_flux_mean_from_geometry",
)
