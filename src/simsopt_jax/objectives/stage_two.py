"""Composable pure-JAX objectives for filamentary Stage-II optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import jax
import jax.numpy as jnp

from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.curve_geometry import (
    curve_geometry_from_spec,
    pair_linking_number_pure,
)
from simsopt_jax.core.curve_kernels import (
    curvature_p_norm_from_kappa_pure,
    curve_curve_distance_penalty_pure,
    curve_surface_distance_penalty_pure,
    kappa_pure,
)
from simsopt_jax.core.field import coil_specs_from_dof_extraction_spec
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral_from_B
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    FixedSurfaceFluxSpec,
    apply_coil_symmetry,
)

from .stochastic_stage_two import (
    StochasticCoilPerturbations,
    _validate_sample_tile,
    stochastic_flux_mean_from_geometry,
)


class CoilDofExtractionProvider(Protocol):
    """Structural contract needed to compose a Stage-II objective."""

    def coil_dof_extraction_spec(self) -> CoilSetDofExtractionSpec: ...


@dataclass(frozen=True, slots=True)
class StageTwoObjectiveConfig:
    """Immutable weights and thresholds for a filamentary Stage-II objective."""

    num_base_curves: int
    length_weight: float = 0.0
    length_target: float | None = None
    length_target_mode: Literal["max", "identity"] = "max"
    individual_length_target: float | None = None
    individual_length_weight: float = 0.0
    curve_curve_minimum_distance: float = 0.1
    curve_curve_weight: float = 0.0
    curve_surface_minimum_distance: float = 0.3
    curve_surface_weight: float = 0.0
    curvature_threshold: float = 5.0
    curvature_weight: float = 0.0
    mean_squared_curvature_threshold: float = 5.0
    mean_squared_curvature_target_mode: Literal["max", "identity"] = "max"
    mean_squared_curvature_weight: float = 0.0
    arclength_variation_weight: float = 0.0
    linking_number_weight: float = 0.0


def _zero(reference: jax.Array) -> jax.Array:
    return jnp.sum(reference[:0])


def stage_two_length_penalty(
    total_length: jax.Array,
    config: StageTwoObjectiveConfig,
    length_weight: float | jax.Array,
) -> jax.Array:
    """Weigh a total base-curve length against the configured length target.

    The caller supplies the weight instead of the config carrying it, so a
    device-resident weight can vary without retracing.  The config still owns
    the target and whether the excess is one-sided.

    ``length_weight`` is authoritative here: ``config.length_weight`` is never
    read by this helper.  Static callers pass ``config.length_weight`` through
    explicitly, and the parametric penalty factory requires
    ``config.length_weight == 0`` so the two spellings cannot disagree.
    """
    if config.length_target is None:
        return length_weight * total_length
    excess = total_length - config.length_target
    if config.length_target_mode == "max":
        excess = jnp.maximum(excess, 0.0)
    return 0.5 * length_weight * excess * excess


def stage_two_linking_number(
    gamma: jax.Array,
    gammadash: jax.Array,
) -> jax.Array:
    """Return the total discrete linking number for a stacked coil set."""
    pairs = tuple(
        (first, second)
        for first in range(int(gamma.shape[0]))
        for second in range(first)
    )
    first = jnp.asarray(tuple(pair[0] for pair in pairs), dtype=jnp.int32)
    second = jnp.asarray(tuple(pair[1] for pair in pairs), dtype=jnp.int32)
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
    return jnp.sum(linking_numbers)


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
        result = result + stage_two_length_penalty(
            jnp.sum(lengths),
            config,
            config.length_weight,
        )

    if config.individual_length_weight != 0.0:
        if config.individual_length_target is None:
            raise ValueError(
                "individual_length_target is required when "
                "individual_length_weight is nonzero."
            )
        individual_lengths = jnp.mean(
            jnp.linalg.norm(base_gammadash, axis=2),
            axis=1,
        )
        individual_excess = individual_lengths - config.individual_length_target
        result = result + (
            0.5
            * config.individual_length_weight
            * jnp.sum(individual_excess * individual_excess)
        )

    base_speed = jnp.linalg.norm(base_gammadash, axis=2)

    if config.curvature_weight != 0.0 or config.mean_squared_curvature_weight != 0.0:
        base_kappa = jax.vmap(kappa_pure)(base_gammadash, base_gammadashdash)
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
            excess = mean_squared_curvature - config.mean_squared_curvature_threshold
            if config.mean_squared_curvature_target_mode == "max":
                excess = jnp.maximum(excess, 0.0)
            result = result + (
                0.5 * config.mean_squared_curvature_weight * jnp.sum(excess * excess)
            )

    if config.arclength_variation_weight != 0.0:
        result = result + config.arclength_variation_weight * jnp.sum(
            jnp.var(base_speed, axis=1)
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
        result = result + config.linking_number_weight * stage_two_linking_number(
            gamma,
            gammadash,
        )

    return result


def stage_two_coil_geometry(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    coil_specs = coil_specs_from_dof_extraction_spec(extraction, parameters)
    parameter_sum = jnp.sum(parameters)
    parameter_zero = parameter_sum - parameter_sum
    geometry: list[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = []
    geometry_by_curve: dict[
        int,
        tuple[jax.Array, jax.Array, jax.Array],
    ] = {}
    for coil_spec in coil_specs:
        curve_id = id(coil_spec.curve)
        curve_geometry = geometry_by_curve.get(curve_id)
        if curve_geometry is None:
            curve_geometry = cast(
                tuple[jax.Array, jax.Array, jax.Array],
                curve_geometry_from_spec(coil_spec.curve),
            )
            geometry_by_curve[curve_id] = curve_geometry
        gamma, gammadash, gammadashdash = curve_geometry
        gamma, gammadash, current = apply_coil_symmetry(
            gamma,
            gammadash,
            coil_spec.current.value[0],
            coil_spec.symmetry,
        )
        if coil_spec.symmetry.has_rotation:
            gammadashdash = gammadashdash @ coil_spec.symmetry.rotmat
        geometry.append((gamma, gammadash, gammadashdash, current + parameter_zero))
    gammas, gammadashs, gammadashdashs, currents = zip(
        *geometry,
        strict=True,
    )
    return (
        jnp.stack(gammas),
        jnp.stack(gammadashs),
        jnp.stack(gammadashdashs),
        jnp.stack(currents),
    )


def stage_two_planar_topology_values(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
    num_base_curves: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return planarity, linking, and physical coordinates for planar coils."""
    gamma, gammadash, _, _ = stage_two_coil_geometry(extraction, parameters)
    base_gamma = gamma[:num_base_curves]
    centered = base_gamma - jnp.mean(base_gamma, axis=1, keepdims=True)
    covariance = jnp.einsum("nqi,nqj->nij", centered, centered) / base_gamma.shape[1]
    minimum_variance = jnp.linalg.eigvalsh(covariance)[:, 0]
    lengths = jnp.mean(
        jnp.linalg.norm(gammadash[:num_base_curves], axis=2),
        axis=1,
    )
    canonical_geometry = jnp.concatenate(
        (
            lengths[:, None],
            jnp.mean(base_gamma, axis=1),
            covariance.reshape((base_gamma.shape[0], 9)),
        ),
        axis=1,
    ).reshape((-1,))
    return (
        jnp.sum(jnp.square(minimum_variance)),
        stage_two_linking_number(gamma, gammadash),
        canonical_geometry,
    )


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
        gamma, gammadash, gammadashdash, _ = stage_two_coil_geometry(
            extraction,
            parameters,
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


def fused_stage_two_values(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate Stage-II objective and diagnostics from one coil geometry pass."""
    gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
        extraction,
        parameters,
    )
    magnetic_field = biot_savart_B(
        flux_spec.points,
        gamma,
        gammadash,
        currents,
    )
    squared_flux = fixed_surface_flux_integral_from_B(
        magnetic_field,
        flux_spec,
    )
    geometric_penalty = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        surface_gamma,
        surface_normal,
        config,
    )
    base_speed = jnp.linalg.norm(
        gammadash[: config.num_base_curves],
        axis=2,
    )
    total_curve_length = jnp.sum(jnp.mean(base_speed, axis=1))
    surface_normal_flat = flux_spec.normal.reshape((-1, 3))
    unit_normal = surface_normal_flat / jnp.linalg.norm(
        surface_normal_flat,
        axis=1,
        keepdims=True,
    )
    maximum_normal_field = jnp.max(
        jnp.abs(jnp.sum(magnetic_field * unit_normal, axis=1))
    )
    return (
        squared_flux + geometric_penalty,
        squared_flux,
        geometric_penalty,
        maximum_normal_field,
        total_curve_length,
    )


def make_fused_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> Callable[[jax.Array], jax.Array]:
    """Compose Stage II without duplicating coil geometry evaluation."""
    extraction = field.coil_dof_extraction_spec()

    def objective(parameters: jax.Array) -> jax.Array:
        return fused_stage_two_values(
            extraction,
            parameters,
            flux_spec,
            surface_gamma,
            surface_normal,
            config,
        )[0]

    return objective


def make_stochastic_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    perturbations: StochasticCoilPerturbations,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
    *,
    sample_tile: int | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Compose a scan-based stochastic flux mean with nominal coil penalties.

    ``sample_tile`` selects the sample-axis lever of
    ``stochastic_flux_mean_from_geometry`` and is validated here, at build
    time, against this bundle's sample count; ``None`` keeps the sequential
    scan oracle. Callers build one objective per perturbation bundle, so the
    same lever also reaches the 256-sample out-of-sample bundle, where a tile
    of ``t`` multiplies peak memory by ``t`` -- that lane stays scanned.
    """
    _validate_sample_tile(sample_tile, perturbations.gamma.shape[0])
    extraction = field.coil_dof_extraction_spec()

    def objective(parameters: jax.Array) -> jax.Array:
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            extraction,
            parameters,
        )
        stochastic_flux = stochastic_flux_mean_from_geometry(
            gamma,
            gammadash,
            currents,
            flux_spec,
            perturbations,
            sample_tile=sample_tile,
        )
        return stochastic_flux + stage_two_geometric_penalty(
            gamma,
            gammadash,
            gammadashdash,
            surface_gamma,
            surface_normal,
            config,
        )

    return objective
