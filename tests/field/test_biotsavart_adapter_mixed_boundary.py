"""Mixed-compute contracts for per-coil Biot-Savart adapter kernels."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.core.field import grouped_coil_set_spec_from_lists
from simsopt_jax_adapters.field import biotsavart_backend as _backend


@pytest.mark.parametrize("batch_size", [0, 1])
def test_per_coil_boundary_stages_inputs_but_preserves_kernel_output(
    monkeypatch,
    batch_size,
):
    monkeypatch.setattr(
        _backend,
        "_as_compute_array",
        lambda value: jnp.asarray(value, dtype=jnp.float32),
    )
    points = jnp.asarray(
        [[0.2, 0.1, -0.3], [0.1, -0.4, 0.0]],
        dtype=jnp.float64,
    )
    gammas = [
        jnp.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=jnp.float64,
        ),
        jnp.asarray(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=jnp.float64,
        ),
    ]
    gammadashs = [jnp.ones_like(gamma) for gamma in gammas]
    currents = [
        jnp.asarray([2.0], dtype=jnp.float64),
        jnp.asarray([3.0], dtype=jnp.float64),
    ]
    coil_set_spec = grouped_coil_set_spec_from_lists(
        gammas,
        gammadashs,
        currents,
    )
    observed_dtypes = []

    def kernel(kernel_points, gamma, gammadash, current):
        observed_dtypes.append(
            (
                kernel_points.dtype,
                gamma.dtype,
                gammadash.dtype,
                current.dtype,
            )
        )
        return jnp.zeros((kernel_points.shape[0], 3), dtype=jnp.float64)

    result = _backend._per_coil_unit_field_with_batch_size(
        points,
        coil_set_spec,
        kernel,
        batch_size=batch_size,
    )

    assert observed_dtypes
    assert all(
        dtypes == (jnp.float32, jnp.float32, jnp.float32, jnp.float32)
        for dtypes in observed_dtypes
    )
    assert len(result) == 2
    assert all(field.dtype == jnp.float64 for field in result)
    for field in result:
        np.testing.assert_array_equal(field, np.zeros((2, 3), dtype=np.float64))
