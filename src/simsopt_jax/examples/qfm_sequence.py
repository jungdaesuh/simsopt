"""Device-resident three-label QFM surface optimization workflow."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.core.qfm_solver import (
    QfmAugmentedLagrangianInfo,
    QfmPenaltySolveInfo,
    qfm_augmented_lagrangian_solve_jax,
    qfm_label_jax_from_dofs,
    qfm_penalty_solve_jax,
    qfm_residual_jax_from_dofs,
)
from simsopt_jax.core.specs import GroupedCoilSetSpec, SurfaceRZFourierSpec
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs


class QfmDeviceState(NamedTuple):
    """QFM value, label value, and their surface-DOF derivatives."""

    parameters: jax.Array
    qfm_value: jax.Array
    qfm_gradient: jax.Array
    label_value: jax.Array
    label_gradient: jax.Array
    label_residual_abs: jax.Array


class QfmStageDeviceResult(NamedTuple):
    """One penalty-then-exact QFM stage whose arrays remain on device."""

    target: jax.Array
    initial: QfmDeviceState
    penalty: QfmDeviceState
    exact: QfmDeviceState
    penalty_optimizer: QfmPenaltySolveInfo
    exact_optimizer: QfmAugmentedLagrangianInfo
    volume_persistence_objective: jax.Array


class QfmSequenceDeviceResult(NamedTuple):
    """The volume, toroidal-flux, and area stages from the native example."""

    initial_volume: jax.Array
    volume: QfmStageDeviceResult
    toroidal_flux: QfmStageDeviceResult
    area: QfmStageDeviceResult


@partial(jax.jit, static_argnames=("label", "toroidal_flux_idx"))
def _qfm_state(
    parameters: jax.Array,
    surface_spec: SurfaceRZFourierSpec,
    coil_set_spec: GroupedCoilSetSpec,
    target: jax.Array,
    *,
    label: str,
    toroidal_flux_idx: int,
) -> QfmDeviceState:
    def values(surface_dofs: jax.Array) -> jax.Array:
        return jnp.stack(
            (
                qfm_residual_jax_from_dofs(
                    surface_spec,
                    surface_dofs,
                    coil_set_spec,
                ),
                qfm_label_jax_from_dofs(
                    surface_spec,
                    surface_dofs,
                    coil_set_spec,
                    label=label,
                    toroidal_flux_idx=toroidal_flux_idx,
                ),
            )
        )

    def values_with_aux(surface_dofs: jax.Array) -> tuple[jax.Array, jax.Array]:
        state_values = values(surface_dofs)
        return state_values, state_values

    state_jacobian, state_values = jax.jacrev(
        values_with_aux,
        has_aux=True,
    )(parameters)
    return QfmDeviceState(
        parameters=parameters,
        qfm_value=state_values[0],
        qfm_gradient=state_jacobian[0],
        label_value=state_values[1],
        label_gradient=state_jacobian[1],
        label_residual_abs=jnp.abs(state_values[1] - target),
    )


@partial(jax.jit, static_argnames=("label", "toroidal_flux_idx"))
def _label_value(
    parameters: jax.Array,
    surface_spec: SurfaceRZFourierSpec,
    coil_set_spec: GroupedCoilSetSpec,
    *,
    label: str,
    toroidal_flux_idx: int,
) -> jax.Array:
    return qfm_label_jax_from_dofs(
        surface_spec,
        parameters,
        coil_set_spec,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )


@jax.jit
def _volume_persistence_objective(
    parameters: jax.Array,
    surface_spec: SurfaceRZFourierSpec,
    coil_set_spec: GroupedCoilSetSpec,
    initial_volume: jax.Array,
) -> jax.Array:
    volume = qfm_label_jax_from_dofs(
        surface_spec,
        parameters,
        coil_set_spec,
        label="volume",
    )
    residual = volume - initial_volume
    return jnp.asarray(0.5, dtype=residual.dtype) * residual * residual


def _solve_stage(
    *,
    initial_parameters: jax.Array,
    surface_spec: SurfaceRZFourierSpec,
    coil_set_spec: GroupedCoilSetSpec,
    label: str,
    toroidal_flux_idx: int,
    initial_volume: jax.Array,
    max_steps: int,
    tolerance: float,
    constraint_weight: float,
) -> QfmStageDeviceResult:
    target = _label_value(
        initial_parameters,
        surface_spec,
        coil_set_spec,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    initial = _qfm_state(
        initial_parameters,
        surface_spec,
        coil_set_spec,
        target,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    penalty_parameters, penalty_optimizer = qfm_penalty_solve_jax(
        surface_spec,
        coil_set_spec,
        label,
        target,
        constraint_weight,
        initial_parameters,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_iter=max_steps,
        tol=tolerance,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    penalty = _qfm_state(
        penalty_parameters,
        surface_spec,
        coil_set_spec,
        target,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    max_outer = min(10, max_steps)
    exact_parameters, exact_optimizer = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        label,
        target,
        penalty_parameters,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=max_outer,
        inner_max_iter=max(1, max_steps // max_outer),
        tol=tolerance,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    exact = _qfm_state(
        exact_parameters,
        surface_spec,
        coil_set_spec,
        target,
        label=label,
        toroidal_flux_idx=toroidal_flux_idx,
    )
    return QfmStageDeviceResult(
        target=target,
        initial=initial,
        penalty=penalty,
        exact=exact,
        penalty_optimizer=penalty_optimizer,
        exact_optimizer=exact_optimizer,
        volume_persistence_objective=_volume_persistence_objective(
            exact_parameters,
            surface_spec,
            coil_set_spec,
            initial_volume,
        ),
    )


def solve_qfm_sequence(
    *,
    initial_parameters: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    coil_set_spec: GroupedCoilSetSpec,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    max_steps: int,
    tolerance: float,
    constraint_weight: float = 1.0,
) -> QfmSequenceDeviceResult:
    """Run the native example's three QFM stages with one final host boundary."""
    parameters = jnp.asarray(initial_parameters, dtype=jnp.float64)
    surface_spec = surface_rz_fourier_spec_from_dofs(
        parameters,
        quadpoints_phi=jnp.asarray(quadpoints_phi, dtype=jnp.float64),
        quadpoints_theta=jnp.asarray(quadpoints_theta, dtype=jnp.float64),
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    initial_volume = _label_value(
        parameters,
        surface_spec,
        coil_set_spec,
        label="volume",
        toroidal_flux_idx=0,
    )
    volume = _solve_stage(
        initial_parameters=parameters,
        surface_spec=surface_spec,
        coil_set_spec=coil_set_spec,
        label="volume",
        toroidal_flux_idx=0,
        initial_volume=initial_volume,
        max_steps=max_steps,
        tolerance=tolerance,
        constraint_weight=constraint_weight,
    )
    toroidal_flux = _solve_stage(
        initial_parameters=volume.exact.parameters,
        surface_spec=surface_spec,
        coil_set_spec=coil_set_spec,
        label="toroidal_flux",
        toroidal_flux_idx=0,
        initial_volume=initial_volume,
        max_steps=max_steps,
        tolerance=tolerance,
        constraint_weight=constraint_weight,
    )
    area = _solve_stage(
        initial_parameters=toroidal_flux.exact.parameters,
        surface_spec=surface_spec,
        coil_set_spec=coil_set_spec,
        label="area",
        toroidal_flux_idx=0,
        initial_volume=initial_volume,
        max_steps=max_steps,
        tolerance=tolerance,
        constraint_weight=constraint_weight,
    )
    return jax.block_until_ready(
        QfmSequenceDeviceResult(
            initial_volume=initial_volume,
            volume=volume,
            toroidal_flux=toroidal_flux,
            area=area,
        )
    )
