"""Strict-transfer coverage for the shared RZ curve-length workflow."""

from __future__ import annotations

import jax
import numpy as np
from simsopt.geo import CurveRZFourier
from simsopt_jax.examples import solve_rz_curve_length


def test_rz_curve_length_solve_has_no_numerical_host_to_device_transfers() -> None:
    curve = CurveRZFourier(100, 4, 5, True)
    initial_full = np.random.RandomState(0).rand(curve.dof_size) - 0.5
    initial_full[0] = 3.0
    curve.x = initial_full
    curve.fix(0)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    full_dofs_device = jax.device_put(
        np.asarray(curve.local_full_x, dtype=np.float64)
    )
    quadpoints_device = jax.device_put(
        np.asarray(curve.quadpoints, dtype=np.float64)
    )
    free_positions_device = jax.device_put(free_positions)

    with jax.transfer_guard("disallow"):
        result = solve_rz_curve_length(
            full_dofs=full_dofs_device,
            quadpoints=quadpoints_device,
            free_positions=free_positions_device,
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
            max_steps=512,
            rtol=1.0e-10,
            atol=1.0e-8,
        )

    expected_length = 6.0 * np.pi
    np.testing.assert_allclose(
        jax.device_get(result.final_length),
        expected_length,
        rtol=1.0e-9,
        atol=1.0e-9,
    )
