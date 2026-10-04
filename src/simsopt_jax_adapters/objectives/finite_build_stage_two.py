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
from math import isfinite
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax._validation import is_integral
from simsopt_jax.backend.dtypes import runtime_device_put_tree
from simsopt_jax.core import (
    CoilSetDofExtractionSpec,
    CurveFilamentSpec,
    apply_coil_symmetry,
    build_filament_gammas,
    coil_specs_from_dof_extraction_spec,
    curve_geometry_from_dofs,
)
from simsopt_jax.core._pairwise_reductions import pairwise_min_distance_pure
from simsopt_jax.core.curve_geometry import (
    optimizable_input_dofs_from_map_spec,
)
from simsopt_jax.core.curve_kernels import curve_curve_distance_penalty_pure
from simsopt_jax.core.field import grouped_coil_set_spec_from_lists
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral
from simsopt_jax.core.specs import FixedSurfaceFluxSpec, GroupedCoilSetSpec
from simsopt_jax.objectives import CoilDofExtractionProvider
from simsopt_jax.pytree import pytree_dataclass
from simsopt_jax.runtime.host_boundary import host_value


@pytree_dataclass(
    data=(
        "filament_offsets",
        "length_targets",
        "length_weight",
        "curve_curve_minimum_distance",
        "curve_curve_weight",
    ),
    meta=("num_base_curves", "symmetry_copies"),
)
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


def _prepare_finite_build_config(
    config: FiniteBuildStageTwoConfig,
    extraction: CoilSetDofExtractionSpec,
) -> FiniteBuildStageTwoConfig:
    """Validate the symmetry-major pack layout used by the geometry kernel."""
    for name in ("num_base_curves", "symmetry_copies"):
        value = getattr(config, name)
        if not is_integral(value) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    host_config = host_value(config)
    if not host_config.filament_offsets:
        raise ValueError("filament_offsets must contain at least one filament.")
    if len(config.length_targets) != config.num_base_curves:
        raise ValueError("length_targets must contain one target per base curve.")
    for offset in host_config.filament_offsets:
        if len(offset) != 2 or not all(isfinite(value) for value in offset):
            raise ValueError("Each filament offset must contain two finite values.")
    for value in (
        *host_config.length_targets, host_config.length_weight,
        host_config.curve_curve_minimum_distance, host_config.curve_curve_weight,
    ):
        if not isfinite(value):
            raise ValueError("Finite-build targets, weights, and thresholds must be finite.")
    pack_count = config.num_base_curves * config.symmetry_copies
    if len(extraction.coils) != pack_count * config.filaments_per_base:
        raise ValueError("Finite-build topology does not match the extracted coil count.")
    shapes = {coil.curve.quadpoints.shape for coil in extraction.coils}
    if len(shapes) != 1 or not next(iter(shapes))[0]:
        raise ValueError("Finite-build coils require matching nonempty quadrature grids.")
    # Compare the frozen templates once; computation reuses the leading pack's
    # base curve/frame and each symmetry pack's representative current.
    coils = host_value(extraction).coils
    for index, coil in enumerate(coils):
        if not isinstance(coil.curve, CurveFilamentSpec):
            raise ValueError("Finite-build objectives require filament curve specs.")
        filament_index = index % config.filaments_per_base
        pack_start = index - filament_index
        representative = coils[pack_start]
        base_index = (index // config.filaments_per_base) % config.num_base_curves
        base = coils[base_index * config.filaments_per_base]
        base_filament = cast(CurveFilamentSpec, base.curve)
        if (coil.curve.dn, coil.curve.db) != host_config.filament_offsets[filament_index]:
            raise ValueError("filament_offsets do not match the extracted filament order.")
        for actual, expected in (
            (coil.curve.base_curve, base_filament.base_curve),
            (coil.curve.base_curve_map, base_filament.base_curve_map),
            (coil.curve.rotation, base_filament.rotation),
            (coil.curve.rotation_map, base_filament.rotation_map),
            (coil.curve.dofs, base_filament.dofs),
            (coil.curve.quadpoints, base_filament.quadpoints),
            (coil.curve_map, base.curve_map),
            (coil.symmetry, representative.symmetry),
            (coil.current_map, representative.current_map),
            (coil.current_term_maps, representative.current_term_maps),
            (coil.current_term_scales, representative.current_term_scales),
        ):
            actual_leaves, actual_tree = jax.tree.flatten(actual)
            expected_leaves, expected_tree = jax.tree.flatten(expected)
            if actual_tree != expected_tree or not all(
                np.array_equal(a, b)
                for a, b in zip(actual_leaves, expected_leaves, strict=True)
            ):
                raise ValueError(
                    "Finite-build filaments must share their base geometry, "
                    "frame, and pack current."
                )
        if coil.curve.frame_kind != base_filament.frame_kind:
            raise ValueError("Finite-build filaments must share their frame kind.")
    return runtime_device_put_tree(config, preserve_placement=True)


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


def _symmetric_pair_indices(count: int) -> tuple[jax.Array, jax.Array]:
    pairs = tuple((first, second) for first in range(count) for second in range(first))
    return (
        jnp.asarray(tuple(pair[0] for pair in pairs), dtype=jnp.int32),
        jnp.asarray(tuple(pair[1] for pair in pairs), dtype=jnp.int32),
    )


# Packing order of the diagnostics vector; coil lengths follow these entries.
# Consumers index the packed array through this tuple, never by literal.
FINITE_BUILD_DIAGNOSTIC_FIELDS = (
    "squared_flux",
    "length_penalty",
    "distance_penalty",
    "minimum_clearance",
)


def _finite_build_penalties(
    base_gammas: jax.Array,
    base_gammadashs: jax.Array,
    config: FiniteBuildStageTwoConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    base_lengths = jnp.mean(
        jnp.linalg.norm(base_gammadashs[: config.num_base_curves], axis=-1),
        axis=1,
    )
    targets = jnp.asarray(config.length_targets, dtype=base_lengths.dtype)
    length_excess = jnp.maximum(base_lengths - targets, 0.0)
    length_penalty = 0.5 * config.length_weight * jnp.sum(length_excess * length_excess)

    first, second = _symmetric_pair_indices(int(base_gammas.shape[0]))
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
    return length_penalty, distance_penalty, base_lengths


def _minimum_clearance(base_gammas: jax.Array) -> jax.Array:
    """Diagnostics-only reduction: never part of the repeated objective."""
    first, second = _symmetric_pair_indices(int(base_gammas.shape[0]))
    return jnp.min(
        jax.vmap(pairwise_min_distance_pure)(
            base_gammas[first],
            base_gammas[second],
        )
    )


def make_finite_build_stage_two_objective(
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    config: FiniteBuildStageTwoConfig,
):
    """Compose the native finite-build flux, length, and clearance terms."""
    extraction = field.coil_dof_extraction_spec()
    config = _prepare_finite_build_config(config, extraction)

    def objective(parameters: jax.Array) -> jax.Array:
        coil_set, base_gammas, base_gammadashs = _finite_build_geometry(
            extraction,
            parameters,
            config,
        )
        length_penalty, distance_penalty, _lengths = _finite_build_penalties(
            base_gammas,
            base_gammadashs,
            config,
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
    config = _prepare_finite_build_config(config, extraction)
    if config.num_base_curves * config.symmetry_copies < 2:
        raise ValueError("Finite-build clearance requires at least two coil packs.")

    def diagnostics(parameters: jax.Array) -> jax.Array:
        coil_set, base_gammas, base_gammadashs = _finite_build_geometry(
            extraction,
            parameters,
            config,
        )
        length_penalty, distance_penalty, lengths = _finite_build_penalties(
            base_gammas,
            base_gammadashs,
            config,
        )
        diagnostics_by_field = {
            "squared_flux": fixed_surface_flux_integral(coil_set, flux_spec),
            "length_penalty": length_penalty,
            "distance_penalty": distance_penalty,
            "minimum_clearance": _minimum_clearance(base_gammas),
        }
        if set(diagnostics_by_field) != set(FINITE_BUILD_DIAGNOSTIC_FIELDS):
            raise ValueError(
                "diagnostics packing drifted from FINITE_BUILD_DIAGNOSTIC_FIELDS"
            )
        return jnp.concatenate(
            (
                jnp.stack(
                    tuple(
                        diagnostics_by_field[name]
                        for name in FINITE_BUILD_DIAGNOSTIC_FIELDS
                    )
                ),
                lengths,
            )
        )

    return diagnostics


__all__ = (
    "FINITE_BUILD_DIAGNOSTIC_FIELDS",
    "FiniteBuildStageTwoConfig",
    "finite_build_stage_two_diagnostics",
    "make_finite_build_stage_two_objective",
)
