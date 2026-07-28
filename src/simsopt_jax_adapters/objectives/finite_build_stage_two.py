"""Traceable finite-build multifilament Stage-II objectives.

The native workflow represents each coil pack with several filaments that
share one base curve and one Fourier frame rotation.  This module preserves
that ownership in the compiled program: each base curve and rotated frame is
evaluated once, then all filament offsets and stellarator-symmetry copies are
materialized from those arrays.  The resulting field, length penalties, and
coil-clearance penalty remain on the selected JAX device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from simsopt_jax.core import (
    CoilSetDofExtractionSpec,
    CurveFilamentSpec,
    apply_coil_symmetry,
    build_filament_gammas,
    coil_specs_from_dof_extraction_spec,
    curve_geometry_from_dofs,
)
from simsopt_jax.core.curve_geometry import (
    optimizable_input_dofs_from_map_spec,
)
from simsopt_jax.core.curve_kernels import curve_curve_distance_penalty_pure
from simsopt_jax.core._pairwise_reductions import pairwise_min_distance_pure
from simsopt_jax.core.field import grouped_coil_set_spec_from_lists
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral
from simsopt_jax.core.specs import FixedSurfaceFluxSpec, GroupedCoilSetSpec
from simsopt_jax.objectives import CoilDofExtractionProvider


@dataclass(frozen=True, slots=True)
class FiniteBuildStageTwoConfig:
    """Immutable topology, targets, and weights for a finite-build objective."""

    num_base_curves: int
    filament_offsets: tuple[tuple[float, float], ...]
    symmetry_copies: int
    length_targets: tuple[float, ...]
    length_weight: float
    curve_curve_minimum_distance: float
    curve_curve_weight: float

    @property
    def filaments_per_base(self) -> int:
        return len(self.filament_offsets)


def _base_geometry(
    filament_spec: CurveFilamentSpec,
) -> tuple[jax.Array, jax.Array]:
    base_dofs = optimizable_input_dofs_from_map_spec(
        filament_spec.base_curve_map,
        filament_spec.dofs,
    )
    gamma, gammadash, _gammadashdash = curve_geometry_from_dofs(
        filament_spec.base_curve,
        base_dofs,
    )
    return gamma, gammadash


def _finite_build_geometry(
    extraction: CoilSetDofExtractionSpec,
    parameters: jax.Array,
    config: FiniteBuildStageTwoConfig,
) -> tuple[GroupedCoilSetSpec, jax.Array, jax.Array]:
    coil_specs = coil_specs_from_dof_extraction_spec(extraction, parameters)
    filaments_per_base = config.filaments_per_base
    coils_per_symmetry = config.num_base_curves * filaments_per_base

    base_filaments: list[tuple[tuple[jax.Array, jax.Array], ...]] = []
    base_geometry: list[tuple[jax.Array, jax.Array]] = []
    for base_index in range(config.num_base_curves):
        representative_index = base_index * filaments_per_base
        filament_spec = cast(
            CurveFilamentSpec,
            coil_specs[representative_index].curve,
        )
        base_filaments.append(
            build_filament_gammas(
                filament_spec,
                config.filament_offsets,
                dofs=filament_spec.dofs,
            )
        )
        base_geometry.append(_base_geometry(filament_spec))

    gammas: list[jax.Array] = []
    gammadashs: list[jax.Array] = []
    currents: list[jax.Array] = []
    symmetric_base_gammas: list[jax.Array] = []
    symmetric_base_gammadashs: list[jax.Array] = []
    for symmetry_index in range(config.symmetry_copies):
        symmetry_offset = symmetry_index * coils_per_symmetry
        for base_index in range(config.num_base_curves):
            representative = coil_specs[
                symmetry_offset + base_index * filaments_per_base
            ]
            base_gamma, base_gammadash = base_geometry[base_index]
            symmetric_gamma, symmetric_gammadash, _current = apply_coil_symmetry(
                base_gamma,
                base_gammadash,
                representative.current.value[0],
                representative.symmetry,
            )
            symmetric_base_gammas.append(symmetric_gamma)
            symmetric_base_gammadashs.append(symmetric_gammadash)
            for filament_gamma, filament_gammadash in base_filaments[base_index]:
                gamma, gammadash, current = apply_coil_symmetry(
                    filament_gamma,
                    filament_gammadash,
                    representative.current.value[0],
                    representative.symmetry,
                )
                gammas.append(gamma)
                gammadashs.append(gammadash)
                currents.append(current)

    return (
        grouped_coil_set_spec_from_lists(gammas, gammadashs, currents),
        jnp.stack(symmetric_base_gammas),
        jnp.stack(symmetric_base_gammadashs),
    )


def _finite_build_penalties(
    base_gammas: jax.Array,
    base_gammadashs: jax.Array,
    config: FiniteBuildStageTwoConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    base_lengths = jnp.mean(
        jnp.linalg.norm(base_gammadashs[: config.num_base_curves], axis=-1),
        axis=1,
    )
    targets = jnp.asarray(config.length_targets, dtype=base_lengths.dtype)
    length_excess = jnp.maximum(base_lengths - targets, 0.0)
    length_penalty = (
        0.5 * config.length_weight * jnp.sum(length_excess * length_excess)
    )

    pairs = tuple(
        (first, second)
        for first in range(int(base_gammas.shape[0]))
        for second in range(first)
    )
    first = jnp.asarray(tuple(pair[0] for pair in pairs), dtype=jnp.int32)
    second = jnp.asarray(tuple(pair[1] for pair in pairs), dtype=jnp.int32)
    distances = jax.vmap(
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
        base_gammas[first],
        base_gammadashs[first],
        base_gammas[second],
        base_gammadashs[second],
    )
    distance_penalty = config.curve_curve_weight * jnp.sum(distances)
    minimum_clearance = jnp.min(
        jax.vmap(pairwise_min_distance_pure)(
            base_gammas[first],
            base_gammas[second],
        )
    )
    return length_penalty, distance_penalty, minimum_clearance, base_lengths


def make_finite_build_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    config: FiniteBuildStageTwoConfig,
):
    """Compose the native finite-build flux, length, and clearance terms."""
    extraction = field.coil_dof_extraction_spec()

    def objective(parameters: jax.Array) -> jax.Array:
        coil_set, base_gammas, base_gammadashs = _finite_build_geometry(
            extraction,
            parameters,
            config,
        )
        length_penalty, distance_penalty, _minimum_clearance, _lengths = (
            _finite_build_penalties(
                base_gammas,
                base_gammadashs,
                config,
            )
        )
        return (
            fixed_surface_flux_integral(coil_set, flux_spec)
            + length_penalty
            + distance_penalty
        )

    return objective


def finite_build_stage_two_diagnostics(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    config: FiniteBuildStageTwoConfig,
):
    """Return flux, penalties, minimum clearance, and coil lengths on device."""
    extraction = field.coil_dof_extraction_spec()

    def diagnostics(parameters: jax.Array) -> jax.Array:
        coil_set, base_gammas, base_gammadashs = _finite_build_geometry(
            extraction,
            parameters,
            config,
        )
        length_penalty, distance_penalty, minimum_clearance, lengths = (
            _finite_build_penalties(
                base_gammas,
                base_gammadashs,
                config,
            )
        )
        squared_flux = fixed_surface_flux_integral(coil_set, flux_spec)
        return jnp.concatenate(
            (
                jnp.stack(
                    (
                        squared_flux,
                        length_penalty,
                        distance_penalty,
                        minimum_clearance,
                    )
                ),
                lengths,
            )
        )

    return diagnostics


__all__ = (
    "FiniteBuildStageTwoConfig",
    "finite_build_stage_two_diagnostics",
    "make_finite_build_stage_two_objective",
)
