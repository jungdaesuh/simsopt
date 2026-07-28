"""Dynamic-surface Stage-II objective contracts for hybrid optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    RegularizedCoil,
    coils_via_symmetries,
)
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import SquaredFlux
from simsopt_jax.objectives.dynamic_surface_stage_two import (
    SurfaceRZFourierDofContract,
    freeze_coil_dof_extraction_spec,
    make_dynamic_surface_stage_two_objective,
)
from simsopt_jax.objectives.stage_two import StageTwoObjectiveConfig
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def _problem() -> tuple[
    SurfaceRZFourierDofContract,
    BiotSavartJAX,
    jax.Array,
    jax.Array,
    SurfaceRZFourier,
    list[Coil] | list[RegularizedCoil],
]:
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        nfp=2,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()
    surface.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surface.fix("rc(0,0)")
    base_curves = create_equally_spaced_curves(
        1,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=1,
        numquadpoints=8,
        use_jax_curve=False,
    )
    current = Current(1.0e5)
    current.fix_all()
    coils = coils_via_symmetries(base_curves, [current], surface.nfp, True)
    field = BiotSavartJAX(coils)
    contract = SurfaceRZFourierDofContract.from_surface(surface)
    return (
        contract,
        field,
        jnp.asarray(field.x, dtype=jnp.float64),
        jnp.asarray(surface.local_x, dtype=jnp.float64),
        surface,
        coils,
    )


def test_dynamic_surface_stage_two_is_jittable_and_differentiates_both_blocks() -> None:
    contract, field, coil_dofs, surface_dofs, _surface, _coils = _problem()
    objective = make_dynamic_surface_stage_two_objective(
        field,
        contract,
        StageTwoObjectiveConfig(
            num_base_curves=1,
            length_weight=1.0e-8,
            individual_length_target=3.3,
            individual_length_weight=0.1,
        ),
        definition="local",
    )
    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))

    value, (coil_gradient, surface_gradient) = value_and_grad(
        coil_dofs,
        surface_dofs,
    )

    assert bool(jnp.isfinite(value))
    assert coil_gradient.shape == coil_dofs.shape
    assert surface_gradient.shape == surface_dofs.shape
    assert bool(jnp.all(jnp.isfinite(coil_gradient)))
    assert bool(jnp.all(jnp.isfinite(surface_gradient)))


def test_dynamic_surface_contract_keeps_closed_constants_host_immutable() -> None:
    contract, _field, _coil_dofs, _surface_dofs, _surface, _coils = _problem()

    assert isinstance(contract.free_indices, tuple)
    assert isinstance(contract.full_dof_template, np.ndarray)
    assert not contract.full_dof_template.flags.writeable
    assert isinstance(contract.quadpoints_phi, np.ndarray)
    assert not contract.quadpoints_phi.flags.writeable
    assert isinstance(contract.quadpoints_theta, np.ndarray)
    assert not contract.quadpoints_theta.flags.writeable


def test_dynamic_surface_contract_freezes_coil_spec_constants_on_host() -> None:
    _contract, field, _coil_dofs, _surface_dofs, _surface, _coils = _problem()
    extraction = freeze_coil_dof_extraction_spec(field)

    array_leaves = tuple(
        leaf
        for leaf in jax.tree.leaves(extraction)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )
    host_array_leaves = tuple(
        leaf for leaf in array_leaves if isinstance(leaf, np.ndarray)
    )
    assert array_leaves
    assert len(host_array_leaves) == len(array_leaves)
    assert all(not leaf.flags.writeable for leaf in host_array_leaves)


def test_dynamic_surface_stage_two_surface_gradient_matches_central_difference() -> (
    None
):
    contract, field, coil_dofs, surface_dofs, _surface, _coils = _problem()
    objective = make_dynamic_surface_stage_two_objective(
        field,
        contract,
        StageTwoObjectiveConfig(num_base_curves=1),
        definition="local",
    )
    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    _value, (_coil_gradient, surface_gradient) = value_and_grad(
        coil_dofs,
        surface_dofs,
    )
    index = 0
    step = 1.0e-6
    plus = surface_dofs.at[index].add(step)
    minus = surface_dofs.at[index].add(-step)
    finite_difference = (objective(coil_dofs, plus) - objective(coil_dofs, minus)) / (
        2.0 * step
    )

    np.testing.assert_allclose(
        np.asarray(surface_gradient[index]),
        np.asarray(finite_difference),
        rtol=2.0e-5,
        atol=1.0e-9,
    )


def test_dynamic_surface_stage_two_matches_native_local_flux_and_mixed_gradient() -> (
    None
):
    contract, field, coil_dofs, surface_dofs, surface, coils = _problem()
    objective = make_dynamic_surface_stage_two_objective(
        field,
        contract,
        StageTwoObjectiveConfig(num_base_curves=1),
        definition="local",
    )
    value, (_coil_gradient, surface_gradient) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1))
    )(coil_dofs, surface_dofs)

    native_field = BiotSavart(coils)
    native_flux = SquaredFlux(surface, native_field, definition="local")
    native_value = native_flux.J()
    normal = surface.normal()
    abs_normal = np.linalg.norm(normal, axis=2)
    magnetic_field = native_field.B().reshape(normal.shape)
    field_spatial_derivative = native_field.dB_by_dX().reshape((*normal.shape, 3))
    unit_normal = normal / abs_normal[:, :, None]
    field_normal = np.sum(magnetic_field * unit_normal, axis=2)
    field_norm = np.linalg.norm(magnetic_field, axis=2)
    field_dot_normal = np.sum(magnetic_field * normal, axis=2)
    derivative_position = (field_normal / field_norm**2)[:, :, None] * np.sum(
        field_spatial_derivative
        * (normal - magnetic_field * (field_dot_normal / field_norm**2)[:, :, None])[
            :, :, None, :
        ],
        axis=3,
    )
    derivative_normal = (field_normal / field_norm**2)[
        :, :, None
    ] * magnetic_field - 0.5 * (field_dot_normal**2 / abs_normal**3 / field_norm**2)[
        :, :, None
    ] * normal
    full_native_gradient = surface.dnormal_by_dcoeff_vjp(
        derivative_normal / (normal.shape[0] * normal.shape[1])
    ) + surface.dgamma_by_dcoeff_vjp(
        derivative_position / (normal.shape[0] * normal.shape[1])
    )
    native_surface_gradient = full_native_gradient[
        np.asarray(surface.local_dofs_free_status, dtype=np.bool_)
    ]

    np.testing.assert_allclose(
        np.asarray(value),
        native_value,
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(surface_gradient),
        native_surface_gradient,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
