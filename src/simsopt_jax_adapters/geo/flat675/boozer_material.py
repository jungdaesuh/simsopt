"""Immutable grid/topology material behind the flat-675 Boozer system.

The material binds a surface template and a coil-DOF extraction to the flat
layout, evaluates a candidate's coil and surface blocks into physical
geometry, and assembles the two-column ``(A, b)`` system whose least-squares
solution is the inner state.  All arithmetic here is float64: the flat-675
port has no proposal precision.

Construction goes through :mod:`.construction` — the material is built from
specs, never from a file record, so the archived bundle and user geometry
reach it the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from simsopt_jax.core.field import (
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_coil_specs,
)
from simsopt_jax.core.specs import (
    CoilSetDofExtractionSpec,
    GroupedCoilSetSpec,
    OptimizableDofMapSpec,
    SurfaceXYZTensorFourierSpec,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs

from ._boozer_arrays import build_flat675_boozer_system_arrays
from .formulation import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    Flat675ContractError,
)
from .policy import Flat675BoozerSystemPolicy


def _map_owner_indices(mapping: OptimizableDofMapSpec) -> tuple[int, ...]:
    owner_indices: list[int] = []
    for owner_start, owner_end, target_start, target_end in mapping.owner_segments:
        if (
            owner_start < 0
            or owner_end < owner_start
            or target_start < 0
            or target_end < target_start
            or owner_end - owner_start != target_end - target_start
            or owner_end > FLAT675_COIL_DOF_COUNT
        ):
            raise Flat675ContractError(
                "flat-675 coil extraction contains an invalid owner segment."
            )
        owner_indices.extend(range(owner_start, owner_end))
    return tuple(owner_indices)


def _coil_extraction_owner_indices(
    extraction: CoilSetDofExtractionSpec,
) -> frozenset[int]:
    owner_indices: list[int] = []
    for coil in extraction.coils:
        owner_indices.extend(_map_owner_indices(coil.curve_map))
        owner_indices.extend(_map_owner_indices(coil.current_map))
        if coil.surface_map is not None:
            owner_indices.extend(_map_owner_indices(coil.surface_map))
    return frozenset(owner_indices)


@dataclass(frozen=True, slots=True)
class Flat675CandidateGeometry:
    """Physical geometry a candidate's coil and surface blocks evaluate to.

    Both the Boozer system and the outer objective terms are functions of this
    geometry, so it is produced once per evaluation and shared.
    """

    surface_gamma: jax.Array
    toroidal_tangent: jax.Array
    poloidal_tangent: jax.Array
    coil_set: GroupedCoilSetSpec


@dataclass(frozen=True, slots=True)
class Flat675BoozerSystem:
    """The two-column design matrix and right-hand side at one candidate."""

    design_matrix: jax.Array
    right_hand_side: jax.Array


@dataclass(frozen=True, slots=True)
class Flat675BoozerMaterial:
    """Fixed surface grid and coil topology shared by every candidate."""

    surface_template: SurfaceXYZTensorFourierSpec
    coil_dof_extraction: CoilSetDofExtractionSpec
    mpol: int
    ntor: int
    nfp: int
    nphi: int
    ntheta: int

    def __post_init__(self) -> None:
        surface_template = self.surface_template
        if not isinstance(surface_template, SurfaceXYZTensorFourierSpec):
            raise Flat675ContractError(
                "flat-675 requires a SurfaceXYZTensorFourier runtime surface."
            )
        if not isinstance(self.coil_dof_extraction, CoilSetDofExtractionSpec):
            raise Flat675ContractError(
                "flat-675 requires a typed coil-DOF extraction spec."
            )
        if surface_template.dofs.shape != (FLAT675_SURFACE_DOF_COUNT,):
            raise Flat675ContractError(
                "flat-675 runtime surface must expose exactly "
                f"{FLAT675_SURFACE_DOF_COUNT} DOFs."
            )
        if surface_template.dofs.dtype != jnp.dtype(jnp.float64):
            raise Flat675ContractError(
                "flat-675 runtime surface DOFs must use float64."
            )
        observed_metadata = (
            surface_template.mpol,
            surface_template.ntor,
            surface_template.nfp,
            int(surface_template.quadpoints_phi.shape[0]),
            int(surface_template.quadpoints_theta.shape[0]),
        )
        if observed_metadata != (
            self.mpol,
            self.ntor,
            self.nfp,
            self.nphi,
            self.ntheta,
        ):
            raise Flat675ContractError(
                "flat-675 runtime grid metadata differs from its surface."
            )
        if _coil_extraction_owner_indices(self.coil_dof_extraction) != frozenset(
            range(FLAT675_COIL_DOF_COUNT)
        ):
            raise Flat675ContractError(
                "flat-675 coil extraction must consume every one of "
                f"{FLAT675_COIL_DOF_COUNT} owner DOFs."
            )


def flat675_candidate_geometry(
    material: Flat675BoozerMaterial,
    coil_coordinates: jax.Array,
    surface_coordinates: jax.Array,
) -> Flat675CandidateGeometry:
    """Evaluate one candidate's coil and surface blocks into geometry.

    Both coordinate blocks stay traced, so everything downstream of this call
    differentiates back to the outer coordinates.
    """
    coil = jnp.asarray(coil_coordinates)
    surface = jnp.asarray(surface_coordinates)
    if coil.shape != (FLAT675_COIL_DOF_COUNT,):
        raise Flat675ContractError(
            f"flat-675 coil coordinates must have shape ({FLAT675_COIL_DOF_COUNT},)."
        )
    if surface.shape != (FLAT675_SURFACE_DOF_COUNT,):
        raise Flat675ContractError(
            "flat-675 surface coordinates must have shape "
            f"({FLAT675_SURFACE_DOF_COUNT},)."
        )
    if coil.dtype != jnp.dtype(jnp.float64) or surface.dtype != jnp.dtype(jnp.float64):
        raise Flat675ContractError(
            "flat-675 geometry evaluation requires float64 inputs."
        )
    gamma, toroidal_tangent, poloidal_tangent = surface_gamma_tangents_from_dofs(
        material.surface_template,
        surface,
    )
    return Flat675CandidateGeometry(
        surface_gamma=gamma,
        toroidal_tangent=toroidal_tangent,
        poloidal_tangent=poloidal_tangent,
        coil_set=grouped_coil_set_spec_from_coil_specs(
            coil_specs_from_dof_extraction_spec(material.coil_dof_extraction, coil)
        ),
    )


def build_flat675_boozer_system(
    geometry: Flat675CandidateGeometry,
    policy: Flat675BoozerSystemPolicy,
) -> Flat675BoozerSystem:
    """Evaluate the candidate's field on its surface and assemble ``(A, b)``."""
    magnetic_field = grouped_biot_savart_B_from_spec(
        geometry.surface_gamma.reshape((-1, 3)),
        geometry.coil_set,
    ).reshape(geometry.surface_gamma.shape)
    design_matrix, right_hand_side = build_flat675_boozer_system_arrays(
        magnetic_field,
        geometry.toroidal_tangent,
        geometry.poloidal_tangent,
        weight_by_inverse_field_magnitude=(policy.weight_by_inverse_field_magnitude),
    )
    return Flat675BoozerSystem(
        design_matrix=design_matrix,
        right_hand_side=right_hand_side,
    )


__all__ = [
    "Flat675BoozerMaterial",
    "Flat675BoozerSystem",
    "Flat675CandidateGeometry",
    "build_flat675_boozer_system",
    "flat675_candidate_geometry",
]
