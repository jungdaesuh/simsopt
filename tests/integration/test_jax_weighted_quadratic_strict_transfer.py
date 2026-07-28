"""Strict-transfer coverage for traceable JAX least-squares closures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.examples import solve_weighted_quadratic


def test_weighted_quadratic_solve_keeps_closure_state_on_device() -> None:
    initial_parameters = jax.device_put(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    )
    targets = jax.device_put(np.asarray((1.0, 2.0, 3.0), dtype=np.float64))
    weights = jax.device_put(np.asarray((1.0, 2.0, 3.0), dtype=np.float64))

    with jax.transfer_guard("disallow"):
        result = solve_weighted_quadratic(
            initial_parameters=initial_parameters,
            targets=targets,
            weights=weights,
            max_steps=32,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    np.testing.assert_allclose(
        jax.device_get(result.final_parameters),
        jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float64),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
