"""Traceable force and vacuum-energy terms for Stage-II coil optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp
from simsopt_jax._validation import is_integral
from simsopt_jax.backend.dtypes import runtime_device_put_tree
from simsopt_jax.core.specs import CoilSetDofExtractionSpec
from simsopt_jax.objectives.stage_two import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    prepare_stage_two_config,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
    stage_two_length_penalty,
)
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_value

from simsopt_jax_adapters.field.force import (
    b2energy_pure,
    curve_force_norms_pure,
)


@pytree_dataclass(
    data=("force_weight", "vacuum_energy_weight", "force_power", "force_threshold"),
    meta=("num_force_coils", "downsample"),
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


def _prepare_force_config(
    config: ForceStageTwoConfig,
    extraction: CoilSetDofExtractionSpec,
    target_quadpoints: jax.Array,
    regularizations: jax.Array,
) -> ForceStageTwoConfig:
    """Check discretization against the coil layout before tracing metrics."""
    coil_count = len(extraction.coils)
    if (
        not is_integral(config.num_force_coils)
        or not 0 < config.num_force_coils <= coil_count
    ):
        raise ValueError("num_force_coils must select a nonempty subset of the coils.")
    if not is_integral(config.downsample) or config.downsample <= 0:
        raise ValueError("downsample must be a positive integer.")
    host_config = host_value(config)
    for name in ("force_weight", "vacuum_energy_weight", "force_power", "force_threshold"):
        if not isfinite(getattr(host_config, name)):
            raise ValueError(f"{name} must be finite.")
    if host_config.force_power <= 0.0:
        raise ValueError("force_power must be positive.")
    shapes = {coil.curve.quadpoints.shape for coil in extraction.coils}
    if len(shapes) != 1 or not next(iter(shapes))[0]:
        raise ValueError("Force coils require matching nonempty quadrature grids.")
    point_count = extraction.coils[0].curve.quadpoints.shape[0]
    if target_quadpoints.shape != (config.num_force_coils, point_count):
        raise ValueError("target_quadpoints must match the force target quadrature grids.")
    if regularizations.shape != (coil_count,):
        raise ValueError("regularizations must contain one value per coil.")
    return runtime_device_put_tree(config)


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
    force_objective = jnp.sum(
        jnp.maximum(force_norms - config.force_threshold, 0.0) ** config.force_power
        * target_speed
    ) / (force_norms.shape[1] * config.force_power)
    vacuum_energy = b2energy_pure(
        gamma,
        gammadash,
        currents,
        config.downsample,
        regularizations,
    )
    return force_objective, jnp.max(force_norms), vacuum_energy


_compiled_force_stage_two_metrics = jax.jit(_force_stage_two_metrics)


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
    stage_two_config = prepare_stage_two_config(
        stage_two_config,
        extraction,
        surface_gamma,
        surface_normal,
    )
    force_config = _prepare_force_config(
        force_config, extraction, target_quadpoints, regularizations,
    )

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


def make_force_stage_two_length_penalty(
    field: CoilDofExtractionProvider,
    stage_two_config: StageTwoObjectiveConfig,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Evaluate the base-curve length penalty for a device-resident weight.

    The returned penalty completes an objective built from the same
    zero-length-weight config, so one compiled graph serves every weight.
    """
    if stage_two_config.length_weight != 0.0:
        raise ValueError(
            "Parametric Stage-II length weights are explicit device parameters; "
            "config length_weight must be zero."
        )
    extraction = field.coil_dof_extraction_spec()
    stage_two_config = prepare_stage_two_config(stage_two_config, extraction)

    def length_penalty(parameters: jax.Array, length_weight: jax.Array) -> jax.Array:
        _, gammadash, _, _ = stage_two_coil_geometry(extraction, parameters)
        lengths = jnp.mean(
            jnp.linalg.norm(
                gammadash[: stage_two_config.num_base_curves],
                axis=2,
            ),
            axis=1,
        )
        return stage_two_length_penalty(
            jnp.sum(lengths),
            stage_two_config,
            length_weight,
        )

    return length_penalty


def force_stage_two_diagnostics(
    field: CoilDofExtractionProvider,
    target_quadpoints: jax.Array,
    regularizations: jax.Array,
    config: ForceStageTwoConfig,
) -> Callable[[jax.Array], jax.Array]:
    """Return force objective, maximum force, and vacuum energy on device."""
    extraction = field.coil_dof_extraction_spec()
    config = _prepare_force_config(config, extraction, target_quadpoints, regularizations)

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
    "make_force_stage_two_length_penalty",
    "make_force_stage_two_objective",
)
