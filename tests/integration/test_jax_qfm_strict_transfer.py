"""Strict-transfer coverage for the exact three-stage QFM workflow."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_qfm_sequence
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_qfm_example_has_one_batched_numerical_host_boundary() -> None:
    source = (REPOSITORY_ROOT / "examples/jax/1_Simple/qfm.py").read_text()

    assert source.count("jax.device_get(") == 1
    assert "QfmSurfaceJAX" not in source
    assert "minimize_qfm_" not in source


def test_qfm_sequence_keeps_all_three_numerical_stages_on_device() -> None:
    _, _, magnetic_axis, nfp, native_field = get_data("ncsx")
    quadrature_phi = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    quadrature_theta = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    field = BiotSavartJAX(native_field.coils)
    device = get_runtime_jax_device()
    initial_parameters = jax.device_put(
        np.asarray(surface.get_dofs(), dtype=np.float64),
        device,
    )
    quadrature_phi_device = jax.device_put(quadrature_phi, device)
    quadrature_theta_device = jax.device_put(quadrature_theta, device)
    coil_set_spec = field.coil_set_spec_from_dofs(
        jax.device_put(np.asarray(field.x, dtype=np.float64), device)
    )

    with jax.transfer_guard("disallow"):
        result = solve_qfm_sequence(
            initial_parameters=initial_parameters,
            quadpoints_phi=quadrature_phi_device,
            quadpoints_theta=quadrature_theta_device,
            coil_set_spec=coil_set_spec,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            max_steps=80,
            tolerance=1.0e-8,
            constraint_weight=1.0,
        )

    initial_qfm, final_qfm, final_feasibility = jax.device_get(
        (
            result.volume.initial.qfm_value,
            result.area.exact.qfm_value,
            result.area.exact.label_residual_abs,
        )
    )
    for stage in (result.volume, result.toroidal_flux, result.area):
        assert bool(stage.penalty_optimizer.success)
        assert bool(stage.exact_optimizer.success)
    assert float(final_qfm) < float(initial_qfm)
    assert float(final_feasibility) <= 1.0e-8
