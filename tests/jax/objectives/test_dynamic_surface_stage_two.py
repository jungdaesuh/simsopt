"""Dynamic-surface Stage-II objective contracts for hybrid optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.objectives.dynamic_surface_stage_two import (
    SurfaceRZFourierDofContract,
    make_dynamic_surface_stage_two_objective,
)
from simsopt_jax.objectives.stage_two import StageTwoObjectiveConfig
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def _problem() -> tuple[
    SurfaceRZFourierDofContract,
    BiotSavartJAX,
    jax.Array,
    jax.Array,
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
    field = BiotSavartJAX(
        coils_via_symmetries(base_curves, [current], surface.nfp, True)
    )
    contract = SurfaceRZFourierDofContract.from_surface(surface)
    return (
        contract,
        field,
        jnp.asarray(field.x, dtype=jnp.float64),
        jnp.asarray(surface.local_x, dtype=jnp.float64),
    )


def test_dynamic_surface_stage_two_is_jittable_and_differentiates_both_blocks() -> None:
    contract, field, coil_dofs, surface_dofs = _problem()
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


def test_dynamic_surface_stage_two_surface_gradient_matches_central_difference() -> None:
    contract, field, coil_dofs, surface_dofs = _problem()
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
