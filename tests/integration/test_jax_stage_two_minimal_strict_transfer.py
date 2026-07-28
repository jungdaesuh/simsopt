"""Strict-transfer coverage for the exact minimal Stage-II workflow."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import solve_minimal_stage_two
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "test_files"


def test_minimal_stage_two_keeps_complete_numerical_workflow_on_device() -> None:
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=4,
        ntheta=4,
    )
    base_curves = create_equally_spaced_curves(
        4,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=2,
        numquadpoints=16,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        True,
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    initial_parameters = jax.device_put(
        np.asarray(field.x, dtype=np.float64)
    )
    taylor_direction = jax.device_put(
        np.random.RandomState(1).uniform(size=initial_parameters.shape)
    )
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
    )

    with jax.transfer_guard("disallow"):
        result = solve_minimal_stage_two(
            field=field,
            flux_spec=flux.fixed_surface_flux_spec(),
            surface_gamma=surface_gamma,
            surface_normal=surface_normal,
            initial_parameters=initial_parameters,
            taylor_direction=taylor_direction,
            num_base_curves=4,
            length_weight=1.0,
            length_target=18.0,
            max_steps=80,
            rtol=1.0e-12,
            atol=1.0e-10,
        )

    final_values = jax.device_get(
        (
            result.initial.objective,
            result.final.objective,
            result.final.objective_gradient,
            result.final.total_curve_length,
        )
    )
    assert result.optimizer.success is True
    assert float(final_values[1]) < float(final_values[0])
    assert np.linalg.norm(final_values[2], ord=np.inf) <= 1.0e-4
    assert float(final_values[3]) <= 19.8
