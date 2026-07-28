"""Strict-transfer coverage for the two-stage RZ-surface workflow."""

from __future__ import annotations

import jax
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.examples import solve_rz_surface_area_volume_sequence


def test_surface_area_volume_sequence_keeps_numerical_state_on_device() -> None:
    quadrature = np.linspace(0.0, 1.0, 32, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.1)
    surface.set_zs(1, 0, 0.1)
    surface.fix_all()
    surface.unfix("rc(1,0)")
    surface.unfix("zs(1,0)")

    full_dofs = jax.device_put(
        np.asarray(surface.local_full_x, dtype=np.float64)
    )
    quadpoints_phi = jax.device_put(
        np.asarray(surface.quadpoints_phi, dtype=np.float64)
    )
    quadpoints_theta = jax.device_put(
        np.asarray(surface.quadpoints_theta, dtype=np.float64)
    )
    free_positions = jax.device_put(
        np.flatnonzero(surface.local_dofs_free_status)
    )
    first_targets = jax.device_put(
        np.asarray((8.0, 0.6), dtype=np.float64)
    )
    second_targets = jax.device_put(
        np.asarray((9.0, 0.8), dtype=np.float64)
    )

    with jax.transfer_guard("disallow"):
        result = solve_rz_surface_area_volume_sequence(
            full_dofs=full_dofs,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            free_positions=free_positions,
            first_targets=first_targets,
            second_targets=second_targets,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            max_steps=64,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert result.first.optimizer.success is True
    assert result.second.optimizer.success is True
    np.testing.assert_allclose(
        jax.device_get(result.first.final_residuals),
        np.zeros(2),
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        jax.device_get(result.second.final_residuals),
        np.zeros(2),
        rtol=0.0,
        atol=1.0e-8,
    )
