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


def _validate_sample_tile(sample_tile: int | None, sample_count: int) -> None:
    """Reject a sample tile that cannot partition ``sample_count`` samples."""
    if sample_tile is None:
        return
    if isinstance(sample_tile, bool) or not isinstance(sample_tile, int):
        raise TypeError(
            f"sample_tile must be a Python int; got {type(sample_tile).__name__}"
        )
    if sample_tile < 1:
        raise ValueError(f"sample_tile must be positive; got {sample_tile}.")
    if sample_count % sample_tile != 0:
        raise ValueError(
            "sample_tile must be a positive divisor of the sample count; "
            f"got sample_tile={sample_tile} for {sample_count} samples"
        )


def stochastic_flux_mean_from_geometry(
    gamma: jax.Array,
    gammadash: jax.Array,
    currents: jax.Array,
    flux_spec: FixedSurfaceFluxSpec,
    perturbations: StochasticCoilPerturbations,
    *,
    sample_tile: int | None = None,
) -> jax.Array:
    """Evaluate a sample mean whose peak memory scales with the tile size.

    ``sample_tile=None`` (the default) keeps the sequential per-sample
    ``scan``, holds one sample's Biot-Savart intermediates live at a time,
    and is the oracle path. ``sample_tile=t`` instead scans over
    ``samples // t`` tiles and vmaps the per-sample flux across each tile,
    trading a factor ``t`` of peak Biot-Savart intermediates for sample-axis
    parallelism; at ``t == samples`` a single tile carries every sample field
    at once, so nothing is streamed.

    Summing more than one sample inside a tile changes the floating-point
    reduction order of the sample sum, so ``sample_tile > 1`` is NOT
    bit-identical to the scan oracle and is gated at the ``native_workflow``
    tolerance bucket of ``simsopt_jax.parity_tolerances``. ``sample_tile=1``
    adds exactly one sample per scan step, in the scan's own reduction order,
    and is measured bit-identical to the default path.
    """
    _validate_sample_tile(sample_tile, perturbations.gamma.shape[0])
    gamma = jnp.asarray(gamma)
    gammadash = jnp.asarray(gammadash, dtype=gamma.dtype)
    currents = jnp.asarray(currents, dtype=gamma.dtype)
    zero = jnp.sum(gamma[:0])

    def sample_flux(
        gamma_perturbation: jax.Array,
        gammadash_perturbation: jax.Array,
    ) -> jax.Array:
        field = biot_savart_B(
            flux_spec.points,
            gamma + gamma_perturbation,
            gammadash + gammadash_perturbation,
            currents,
        )
        return fixed_surface_flux_integral_from_B(field, flux_spec)

    sample_count = perturbations.gamma.shape[0]
    if sample_tile is None:

        def accumulate(
            total: jax.Array,
            sample: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, None]:
            gamma_perturbation, gammadash_perturbation = sample
            return total + sample_flux(gamma_perturbation, gammadash_perturbation), None

        total, _ = jax.lax.scan(
            jax.checkpoint(accumulate),
            zero,
            (perturbations.gamma, perturbations.gammadash),
        )
    else:
        tile_count = sample_count // sample_tile
        tiled_shape = (tile_count, sample_tile, *perturbations.gamma.shape[1:])

        def accumulate_tile(
            total: jax.Array,
            tile: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, None]:
            gamma_perturbation, gammadash_perturbation = tile
            tile_flux = jax.vmap(sample_flux)(
                gamma_perturbation,
                gammadash_perturbation,
            )
            return total + jnp.sum(tile_flux), None

        total, _ = jax.lax.scan(
            jax.checkpoint(accumulate_tile),
            zero,
            (
                perturbations.gamma.reshape(tiled_shape),
                perturbations.gammadash.reshape(tiled_shape),
            ),
        )
    return total / jnp.asarray(sample_count, dtype=total.dtype)


__all__ = (
    "StochasticCoilPerturbations",
    "stochastic_flux_mean_from_geometry",
)
