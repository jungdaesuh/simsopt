"""Pure-JAX Stage-II objective on a dynamically optimized RZ surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral_from_B
from simsopt_jax.core.specs import make_fixed_surface_flux_spec
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_spec_from_dofs,
)

from .stage_two import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
)


class SurfaceRZFourierProvider(Protocol):
    """Host-only structural contract used to freeze an RZ surface layout."""

    local_full_x: np.ndarray
    local_dofs_free_status: np.ndarray
    quadpoints_phi: np.ndarray
    quadpoints_theta: np.ndarray
    mpol: int
    ntor: int
    nfp: int
    stellsym: bool


@dataclass(frozen=True, slots=True)
class SurfaceRZFourierDofContract:
    """Immutable mapping from free host surface DOFs to a full JAX spec."""

    full_dof_template: jax.Array
    free_indices: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    mpol: int
    ntor: int
    nfp: int
    stellsym: bool

    @classmethod
    def from_surface(
        cls,
        surface: SurfaceRZFourierProvider,
    ) -> SurfaceRZFourierDofContract:
        """Freeze the complete surface layout once at the host boundary."""
        free_indices = np.flatnonzero(
            np.asarray(surface.local_dofs_free_status, dtype=np.bool_)
        ).astype(np.int32, copy=False)
        return cls(
            full_dof_template=jnp.asarray(surface.local_full_x, dtype=jnp.float64),
            free_indices=jnp.asarray(free_indices, dtype=jnp.int32),
            quadpoints_phi=jnp.asarray(surface.quadpoints_phi, dtype=jnp.float64),
            quadpoints_theta=jnp.asarray(
                surface.quadpoints_theta,
                dtype=jnp.float64,
            ),
            mpol=int(surface.mpol),
            ntor=int(surface.ntor),
            nfp=int(surface.nfp),
            stellsym=bool(surface.stellsym),
        )

    def full_dofs(self, free_dofs: jax.Array) -> jax.Array:
        """Scatter free values into the immutable full-DOF template."""
        return self.full_dof_template.at[self.free_indices].set(free_dofs)

    def surface_spec(self, free_dofs: jax.Array):
        """Build a traceable RZFourier spec from free surface parameters."""
        return surface_rz_fourier_spec_from_dofs(
            self.full_dofs(free_dofs),
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=self.stellsym,
            use_custom_vjp=True,
        )


def make_dynamic_surface_stage_two_objective(
    field: CoilDofExtractionProvider,
    surface: SurfaceRZFourierDofContract,
    config: StageTwoObjectiveConfig,
    *,
    definition: str = "local",
):
    """Compose dynamic-surface flux and coil penalties without host callbacks."""
    extraction = field.coil_dof_extraction_spec()

    def objective(
        coil_dofs: jax.Array,
        surface_free_dofs: jax.Array,
    ) -> jax.Array:
        surface_spec = surface.surface_spec(surface_free_dofs)
        surface_gamma = surface_rz_fourier_gamma_from_spec(surface_spec)
        surface_normal = surface_rz_fourier_normal_from_spec(surface_spec)
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            extraction,
            coil_dofs,
        )
        field_points = surface_gamma.reshape((-1, 3))
        field_values = biot_savart_B(
            field_points,
            gamma,
            gammadash,
            currents,
        )
        flux_spec = make_fixed_surface_flux_spec(
            points=field_points,
            normal=surface_normal,
            target=jnp.zeros(surface_normal.shape[:2], dtype=surface_normal.dtype),
            definition=definition,
        )
        flux = fixed_surface_flux_integral_from_B(field_values, flux_spec)
        geometry_penalty = stage_two_geometric_penalty(
            gamma,
            gammadash,
            gammadashdash,
            surface_gamma.reshape((-1, 3)),
            surface_normal.reshape((-1, 3)),
            config,
        )
        return flux + geometry_penalty

    return objective


__all__ = (
    "SurfaceRZFourierDofContract",
    "make_dynamic_surface_stage_two_objective",
)
