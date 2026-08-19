"""The jittable flat-675 single-stage objective.

``f(x)`` maps one 675-vector of outer coordinates to the certified eight-term
scalar: the coil and surface blocks build the Boozer system, its two-column
least-squares solution closes ``(iota, G)``, and the vessel block enters as
traced geometry so all 675 coordinates carry gradient.  Everything runs in
float64 and nothing here reads the host, so the whole objective fuses into one
device program.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from simsopt_jax.core.specs import SurfaceRZFourierSpec
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_dofs_from_spec,
    surface_rz_fourier_gamma_from_dofs,
)

from simsopt_jax_adapters.geo.surface_objectives import (
    _traceable_single_stage_outer_term_values,
    _traceable_weighted_single_stage_outer_term_values,
)

from .boozer_material import (
    Flat675BoozerMaterial,
    build_flat675_boozer_system,
    flat675_candidate_geometry,
)
from .formulation import (
    FLAT675_COIL_SLICE,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_SLICE,
    FLAT675_VESSEL_DOF_COUNT,
    FLAT675_VESSEL_SLICE,
    Flat675ContractError,
)
from .policy import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    Flat675BoozerSystemPolicy,
    Flat675ObjectivePolicy,
    flat675_outer_objective_config,
)
from .y_solve import solve_flat675_y_qr

_SURFACE_KIND = "xyztensorfourier"


@dataclass(frozen=True, slots=True)
class Flat675Material:
    """Everything a flat-675 evaluation needs besides the candidate itself."""

    boozer: Flat675BoozerMaterial
    vessel_template: SurfaceRZFourierSpec

    def __post_init__(self) -> None:
        if not isinstance(self.boozer, Flat675BoozerMaterial):
            raise Flat675ContractError("boozer must be Flat675BoozerMaterial.")
        if not isinstance(self.vessel_template, SurfaceRZFourierSpec):
            raise Flat675ContractError("vessel_template must be SurfaceRZFourierSpec.")
        vessel_dofs = surface_rz_fourier_dofs_from_spec(self.vessel_template)
        if vessel_dofs.shape != (FLAT675_VESSEL_DOF_COUNT,):
            raise Flat675ContractError(
                "flat-675 vessel template must expose exactly "
                f"{FLAT675_VESSEL_DOF_COUNT} free DOFs; got {vessel_dofs.shape}."
            )
        if vessel_dofs.dtype != jnp.dtype(jnp.float64):
            raise Flat675ContractError("flat-675 vessel template must use float64.")


def flat675_weighted_terms(
    outer_coordinates: jax.Array,
    *,
    material: Flat675Material,
    objective_policy: Flat675ObjectivePolicy,
    boozer_policy: Flat675BoozerSystemPolicy,
) -> jax.Array:
    """Return the eight weighted term values in canonical assembly order.

    The returned vector is indexed by :data:`FLAT675_OBJECTIVE_TERM_KEYS`; the
    objective is its ordered sum.
    """
    coordinates = jnp.asarray(outer_coordinates)
    if coordinates.shape != (FLAT675_OUTER_DOF_COUNT,):
        raise Flat675ContractError(
            "flat-675 outer coordinates must have shape "
            f"({FLAT675_OUTER_DOF_COUNT},); got {coordinates.shape}."
        )
    coil = coordinates[FLAT675_COIL_SLICE]
    vessel = coordinates[FLAT675_VESSEL_SLICE]
    surface = coordinates[FLAT675_SURFACE_SLICE]

    geometry = flat675_candidate_geometry(material.boozer, coil, surface)
    system = build_flat675_boozer_system(geometry, boozer_policy)
    inner_state = solve_flat675_y_qr(
        system.design_matrix,
        system.right_hand_side,
    ).solution

    surface_template = material.boozer.surface_template
    outer_objective_config = flat675_outer_objective_config(
        objective_policy,
        nfp=material.boozer.nfp,
        vessel_gamma=surface_rz_fourier_gamma_from_dofs(
            material.vessel_template,
            vessel,
        ),
    )
    raw_terms = _traceable_single_stage_outer_term_values(
        jnp.concatenate((surface, inner_state)),
        coil,
        geometry.coil_set,
        quadpoints_phi=surface_template.quadpoints_phi,
        quadpoints_theta=surface_template.quadpoints_theta,
        mpol=surface_template.mpol,
        ntor=surface_template.ntor,
        nfp=surface_template.nfp,
        stellsym=surface_template.stellsym,
        scatter_indices=surface_template.scatter_indices,
        surface_kind=_SURFACE_KIND,
        label_quadpoints_phi=surface_template.quadpoints_phi,
        label_quadpoints_theta=surface_template.quadpoints_theta,
        label_mpol=surface_template.mpol,
        label_ntor=surface_template.ntor,
        label_nfp=surface_template.nfp,
        label_stellsym=surface_template.stellsym,
        label_scatter_indices=surface_template.scatter_indices,
        label_surface_kind=_SURFACE_KIND,
        optimize_G=True,
        weight_inv_modB=boozer_policy.weight_by_inverse_field_magnitude,
        constraint_weight=objective_policy.boozer_constraint_weight,
        targetlabel=objective_policy.boozer_target_label,
        label_type=objective_policy.boozer_label_type.value,
        phi_idx=objective_policy.boozer_label_phi_index,
        iota_target=objective_policy.iota_target,
        surface_quadpoints_phi=surface_template.quadpoints_phi,
        surface_quadpoints_theta=surface_template.quadpoints_theta,
        coil_dof_extraction_spec=material.boozer.coil_dof_extraction,
        outer_objective_config=outer_objective_config,
    )
    weighted_terms = _traceable_weighted_single_stage_outer_term_values(
        raw_terms,
        outer_objective_config=outer_objective_config,
    )
    return jnp.stack(
        tuple(weighted_terms[term_key] for term_key in FLAT675_OBJECTIVE_TERM_KEYS)
    )


def flat675_objective(
    outer_coordinates: jax.Array,
    *,
    material: Flat675Material,
    objective_policy: Flat675ObjectivePolicy,
    boozer_policy: Flat675BoozerSystemPolicy,
) -> jax.Array:
    """Return the certified scalar objective at one 675-vector.

    The terms are added left to right in canonical order rather than reduced:
    floating-point addition is not associative, so the assembly order is part
    of the certified value.
    """
    weighted_terms = flat675_weighted_terms(
        outer_coordinates,
        material=material,
        objective_policy=objective_policy,
        boozer_policy=boozer_policy,
    )
    total = weighted_terms[0]
    for term_index in range(1, len(FLAT675_OBJECTIVE_TERM_KEYS)):
        total = total + weighted_terms[term_index]
    return total


@dataclass(frozen=True, slots=True)
class Flat675Programs:
    """One-argument objective and diagnostics closures over fixed material."""

    objective_fn: Callable[[jax.Array], jax.Array]
    diagnostics_fn: Callable[[jax.Array], jax.Array]


def bind_flat675_programs(
    *,
    material: Flat675Material,
    objective_policy: Flat675ObjectivePolicy,
    boozer_policy: Flat675BoozerSystemPolicy,
) -> Flat675Programs:
    """Bind material and policy so callers pass only the 675-vector."""

    def objective_fn(outer_coordinates: jax.Array) -> jax.Array:
        return flat675_objective(
            outer_coordinates,
            material=material,
            objective_policy=objective_policy,
            boozer_policy=boozer_policy,
        )

    def diagnostics_fn(outer_coordinates: jax.Array) -> jax.Array:
        return flat675_weighted_terms(
            outer_coordinates,
            material=material,
            objective_policy=objective_policy,
            boozer_policy=boozer_policy,
        )

    return Flat675Programs(
        objective_fn=objective_fn,
        diagnostics_fn=diagnostics_fn,
    )


__all__ = [
    "Flat675Material",
    "Flat675Programs",
    "bind_flat675_programs",
    "flat675_objective",
    "flat675_weighted_terms",
]
