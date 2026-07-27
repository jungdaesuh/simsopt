"""Composable pure-JAX objectives for filamentary Stage-II optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from simsopt_jax.core import (
    CoilSetDofExtractionSpec,
    coil_specs_from_dof_extraction_spec,
    curve_geometry_from_spec,
)
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_surface_distance_penalty_pure,
    kappa_pure,
)
from simsopt_jax.core.curve_geometry import pair_linking_number_pure


class CoilDofExtractionProvider(Protocol):
    """Structural contract needed to compose a Stage-II objective."""

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec: ...


@dataclass(frozen=True, slots=True)
class StageTwoObjectiveConfig:
    """Immutable weights and thresholds for a filamentary Stage-II objective."""

    num_base_curves: int
    length_weight: float = 0.0
    length_target: float | None = None
    curve_curve_minimum_distance: float = 0.1
    curve_curve_weight: float = 0.0
    curve_surface_minimum_distance: float = 0.3
    curve_surface_weight: float = 0.0
    curvature_threshold: float = 5.0
    curvature_weight: float = 0.0
    mean_squared_curvature_threshold: float = 5.0
    mean_squared_curvature_weight: float = 0.0
    linking_number_weight: float = 0.0


def _zero(reference: jax.Array) -> jax.Array:
    return jnp.sum(reference[:0])


def _length_penalty(total_length: jax.Array, config: StageTwoObjectiveConfig):
    if config.length_target is None:
        return config.length_weight * total_length
    excess = jnp.maximum(total_length - config.length_target, 0.0)
    return 0.5 * config.length_weight * excess * excess


def stage_two_geometric_penalty(
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> jax.Array:
    """Evaluate vectorized length, clearance, and curvature penalties."""
    base_gammadash = gammadash[: config.num_base_curves]
    base_gammadashdash = gammadashdash[: config.num_base_curves]
    zero = _zero(gamma)
    result = zero

    if config.length_weight != 0.0:
        lengths = jnp.mean(jnp.linalg.norm(base_gammadash, axis=2), axis=1)
        result = result + _length_penalty(jnp.sum(lengths), config)

    if config.curvature_weight != 0.0 or config.mean_squared_curvature_weight != 0.0:
        base_kappa = jax.vmap(kappa_pure)(base_gammadash, base_gammadashdash)
        base_speed = jnp.linalg.norm(base_gammadash, axis=2)
        if config.curvature_weight != 0.0:
            curvature = jax.vmap(
                lambda current_kappa, current_gammadash: (
                    curvature_p_norm_from_kappa_pure(
                        current_kappa,
                        current_gammadash,
                        2.0,
                        config.curvature_threshold,
                    )
                )
            )(base_kappa, base_gammadash)
            result = result + config.curvature_weight * jnp.sum(curvature)
        if config.mean_squared_curvature_weight != 0.0:
            mean_squared_curvature = jnp.sum(
                base_kappa * base_kappa * base_speed,
                axis=1,
            ) / jnp.sum(base_speed, axis=1)
            excess = jnp.maximum(
                mean_squared_curvature - config.mean_squared_curvature_threshold,
                0.0,
            )
            result = result + (
                0.5 * config.mean_squared_curvature_weight * jnp.sum(excess * excess)
            )

    if config.curve_curve_weight != 0.0:
        pairs = tuple(
            (index, base_index)
            for index in range(int(gamma.shape[0]))
            for base_index in range(min(index, config.num_base_curves))
        )
        first = jnp.asarray(tuple(pair[0] for pair in pairs), dtype=jnp.int32)
        second = jnp.asarray(tuple(pair[1] for pair in pairs), dtype=jnp.int32)
        curve_curve = jax.vmap(
            lambda gamma_1, gammadash_1, gamma_2, gammadash_2: (
                curve_curve_distance_penalty_pure(
                    gamma_1,
                    gammadash_1,
                    gamma_2,
                    gammadash_2,
                    config.curve_curve_minimum_distance,
                )
            )
        )(
            gamma[first],
            gammadash[first],
            gamma[second],
            gammadash[second],
        )
        result = result + config.curve_curve_weight * jnp.sum(curve_curve)

    if config.curve_surface_weight != 0.0:
        curve_surface = jax.vmap(
            lambda current_gamma, current_gammadash: (
                curve_surface_distance_penalty_pure(
                    current_gamma,
                    current_gammadash,
                    surface_gamma,
                    surface_normal,
                    config.curve_surface_minimum_distance,
                )
            )
        )(gamma, gammadash)
        result = result + config.curve_surface_weight * jnp.sum(curve_surface)

    if config.linking_number_weight != 0.0:
        linking_pairs = tuple(
            (first, second)
            for first in range(int(gamma.shape[0]))
            for second in range(first)
        )
        first = jnp.asarray(tuple(pair[0] for pair in linking_pairs), dtype=jnp.int32)
        second = jnp.asarray(tuple(pair[1] for pair in linking_pairs), dtype=jnp.int32)
        dphi = jnp.reciprocal(jnp.asarray(gamma.shape[1], dtype=gamma.dtype))
        linking_numbers = jax.vmap(
            lambda gamma_1, gammadash_1, gamma_2, gammadash_2: pair_linking_number_pure(
                gamma_1,
                gammadash_1,
                gamma_2,
                gammadash_2,
                dphi,
                dphi,
            )
        )(
            gamma[first],
            gammadash[first],
            gamma[second],
            gammadash[second],
        )
        result = result + config.linking_number_weight * jnp.sum(linking_numbers)

    return result


def make_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_objective: Callable[[jax.Array], jax.Array],
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> Callable[[jax.Array], jax.Array]:
    """Compose quadratic flux with immutable filamentary geometry penalties."""
    extraction = field.coil_dof_extraction_spec()

    def objective(parameters: jax.Array) -> jax.Array:
        coil_specs = coil_specs_from_dof_extraction_spec(extraction, parameters)
        geometry: list[tuple[jax.Array, jax.Array, jax.Array]] = []
        for coil_spec in coil_specs:
            gamma, gammadash, gammadashdash = curve_geometry_from_spec(coil_spec.curve)
            if coil_spec.symmetry.has_rotation:
                rotation = coil_spec.symmetry.rotmat
                gamma = gamma @ rotation
                gammadash = gammadash @ rotation
                gammadashdash = gammadashdash @ rotation
            geometry.append((gamma, gammadash, gammadashdash))
        gamma, gammadash, gammadashdash = (
            jnp.stack(terms) for terms in zip(*geometry, strict=True)
        )
        return flux_objective(parameters) + stage_two_geometric_penalty(
            gamma,
            gammadash,
            gammadashdash,
            surface_gamma,
            surface_normal,
            config,
        )

    return objective
