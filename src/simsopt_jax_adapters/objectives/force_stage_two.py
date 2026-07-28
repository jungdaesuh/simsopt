"""Traceable force and vacuum-energy terms for Stage-II coil optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.objectives import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
)
from simsopt_jax_adapters.field.force import (
    b2energy_pure,
    curve_force_norms_pure,
)


@dataclass(frozen=True, slots=True)
class ForceStageTwoConfig:
    """Immutable weights and discretization for native-equivalent force terms."""

    num_force_coils: int
    force_weight: float = 0.0
    vacuum_energy_weight: float = 0.0
    force_power: float = 4.0
    force_threshold: float = 0.0
    downsample: int = 1


def _force_stage_two_metrics(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    currents: jax.Array,
    target_quadpoints: jax.Array,
    regularizations: jax.Array,
    config: ForceStageTwoConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    target_count = config.num_force_coils
    coil_count = gamma.shape[0]
    target_gamma = jax.lax.slice_in_dim(gamma, 0, target_count, axis=0)
    source_gamma = jax.lax.slice_in_dim(
        gamma,
        target_count,
        coil_count,
        axis=0,
    )
    target_gammadash = jax.lax.slice_in_dim(
        gammadash,
        0,
        target_count,
        axis=0,
    )
    source_gammadash = jax.lax.slice_in_dim(
        gammadash,
        target_count,
        coil_count,
        axis=0,
    )
    target_gammadashdash = jax.lax.slice_in_dim(
        gammadashdash,
        0,
        target_count,
        axis=0,
    )
    target_currents = jax.lax.slice_in_dim(
        currents,
        0,
        target_count,
        axis=0,
    )
    source_currents = jax.lax.slice_in_dim(
        currents,
        target_count,
        coil_count,
        axis=0,
    )
    target_regularizations = jax.lax.slice_in_dim(
        regularizations,
        0,
        target_count,
        axis=0,
    )
    force_norms = curve_force_norms_pure(
        target_gamma,
        source_gamma,
        target_gammadash,
        source_gammadash,
        target_gammadashdash,
        target_quadpoints,
        target_currents,
        source_currents,
        target_regularizations,
        config.downsample,
    )
    sampled_target_gammadash = jax.lax.slice_in_dim(
        target_gammadash,
        0,
        target_gammadash.shape[1],
        stride=config.downsample,
        axis=1,
    )
    target_speed = jnp.linalg.norm(
        sampled_target_gammadash,
        axis=-1,
    )
    force_objective = (
        jnp.sum(
            jnp.maximum(force_norms - config.force_threshold, 0.0)
            ** config.force_power
            * target_speed
        )
        / (force_norms.shape[1] * config.force_power)
    )
    vacuum_energy = b2energy_pure(
        gamma,
        gammadash,
        currents,
        config.downsample,
        regularizations,
    )
    return force_objective, jnp.max(force_norms), vacuum_energy


_compiled_force_stage_two_metrics = jax.jit(
    _force_stage_two_metrics,
    static_argnames=("config",),
)


def make_force_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_objective: Callable[[jax.Array], jax.Array],
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    target_quadpoints: jax.Array,
    regularizations: jax.Array,
    stage_two_config: StageTwoObjectiveConfig,
    force_config: ForceStageTwoConfig,
) -> Callable[[jax.Array], jax.Array]:
    """Compose flux, engineering, Lorentz-force, and vacuum-energy terms."""
    extraction = field.coil_dof_extraction_spec()

    def objective(parameters: jax.Array) -> jax.Array:
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            extraction,
            parameters,
        )
        force_objective, _, vacuum_energy = _compiled_force_stage_two_metrics(
            gamma,
            gammadash,
            gammadashdash,
            currents,
            target_quadpoints,
            regularizations,
            force_config,
        )
        return (
            flux_objective(parameters)
            + stage_two_geometric_penalty(
                gamma,
                gammadash,
                gammadashdash,
                surface_gamma,
                surface_normal,
                stage_two_config,
            )
            + force_config.force_weight * force_objective
            + force_config.vacuum_energy_weight * vacuum_energy
        )

    return objective


def force_stage_two_diagnostics(
    field: CoilDofExtractionProvider,
    target_quadpoints: jax.Array,
    regularizations: jax.Array,
    config: ForceStageTwoConfig,
) -> Callable[[jax.Array], jax.Array]:
    """Return force objective, maximum force, and vacuum energy on device."""
    extraction = field.coil_dof_extraction_spec()

    def diagnostics(parameters: jax.Array) -> jax.Array:
        geometry = stage_two_coil_geometry(extraction, parameters)
        return jnp.stack(
            _compiled_force_stage_two_metrics(
                *geometry,
                target_quadpoints,
                regularizations,
                config,
            )
        )

    return diagnostics


__all__ = (
    "ForceStageTwoConfig",
    "force_stage_two_diagnostics",
    "make_force_stage_two_objective",
)
