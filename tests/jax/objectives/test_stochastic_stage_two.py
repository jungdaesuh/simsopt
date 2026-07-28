"""Pure-JAX stochastic Stage-II reduction contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

from simsopt_jax.core import make_fixed_surface_flux_spec
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
    stochastic_flux_mean_from_geometry,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import (
    SquaredFluxJAX,
    coil_current_fixed_geometry_flux_jax,
)


def _circle_geometry() -> tuple[jax.Array, jax.Array]:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
    gamma = jnp.stack(
        (jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)), axis=1
    )[None, :, :]
    gammadash = jnp.stack(
        (
            -2.0 * jnp.pi * jnp.sin(angles),
            2.0 * jnp.pi * jnp.cos(angles),
            jnp.zeros_like(angles),
        ),
        axis=1,
    )[None, :, :]
    return gamma, gammadash


def test_zero_perturbation_mean_matches_nominal_flux_through_scan() -> None:
    gamma, gammadash = _circle_geometry()
    currents = jnp.asarray((1.0e5,), dtype=jnp.float64)
    flux_spec = make_fixed_surface_flux_spec(
        points=jnp.asarray(((2.0, 0.0, 0.0),), dtype=jnp.float64),
        normal=jnp.asarray((((1.0, 0.0, 0.0),),), dtype=jnp.float64),
        target=jnp.zeros((1, 1), dtype=jnp.float64),
        definition="quadratic flux",
    )
    perturbations = StochasticCoilPerturbations(
        gamma=jnp.zeros((3, *gamma.shape), dtype=jnp.float64),
        gammadash=jnp.zeros((3, *gammadash.shape), dtype=jnp.float64),
    )

    stochastic = stochastic_flux_mean_from_geometry(
        gamma,
        gammadash,
        currents,
        flux_spec,
        perturbations,
    )
    nominal = coil_current_fixed_geometry_flux_jax(
        flux_spec.points,
        gamma,
        gammadash,
        currents,
        flux_spec,
    )
    jaxpr = jax.make_jaxpr(
        lambda current_gamma: stochastic_flux_mean_from_geometry(
            current_gamma,
            gammadash,
            currents,
            flux_spec,
            perturbations,
        )
    )(gamma)

    np.testing.assert_allclose(stochastic, nominal, rtol=1.0e-12, atol=1.0e-12)
    assert any(equation.primitive.name == "scan" for equation in jaxpr.jaxpr.eqns)


def test_zero_error_composed_objective_matches_deterministic_stage_two() -> None:
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    curve = CurveXYZFourier(16, 2)
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    field = BiotSavartJAX([Coil(curve, Current(1.0e5))])
    flux = SquaredFluxJAX(surface, field)
    zero_errors = StochasticCoilPerturbations(
        gamma=jnp.zeros((2, 1, 16, 3), dtype=jnp.float64),
        gammadash=jnp.zeros((2, 1, 16, 3), dtype=jnp.float64),
    )
    objective = make_stochastic_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        zero_errors,
        jnp.asarray(surface.gamma()).reshape((-1, 3)),
        jnp.asarray(surface.normal()).reshape((-1, 3)),
        StageTwoObjectiveConfig(num_base_curves=1),
    )
    parameters = jnp.asarray(field.x, dtype=jnp.float64)

    actual_value, actual_gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    expected_value, expected_gradient = jax.jit(
        jax.value_and_grad(flux.traceable_objective())
    )(parameters)

    np.testing.assert_allclose(actual_value, expected_value, rtol=1.0e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
