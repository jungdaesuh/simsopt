"""Strict-placement coverage for fused surface-objective differentiation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conftest import enable_strict_parity_backend, parity_default_device
from simsopt_jax.core.surface_fourier_kernels import dofs_to_xyzc
from simsopt_jax_adapters.geo.surface_objectives import (
    _make_cached_strict_scalar_value_and_two_gradients,
)


def test_fused_direct_and_inner_gradients_obey_strict_gpu_transfer_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    scatter_indices = np.asarray((0, 8, 9, 17, 18, 26), dtype=np.int32)

    def objective(coil_dofs, inner_dofs, _optimize_G, _weight_inv_modB):
        xc, yc, zc = dofs_to_xyzc(inner_dofs, scatter_indices, 1, 1)
        return jnp.sum(coil_dofs) + jnp.sum(xc) + jnp.sum(yc) + jnp.sum(zc)

    value_and_gradients = _make_cached_strict_scalar_value_and_two_gradients(
        objective
    )
    with parity_default_device("gpu"):
        device = jax.devices("gpu")[0]
        coil_dofs = jax.device_put(
            np.asarray((0.25, -0.5), dtype=np.float64),
            device,
        )
        inner_dofs = jax.device_put(
            np.asarray((1.0, -2.0, 3.0, -4.0, 5.0, -6.0), dtype=np.float64),
            device,
        )

        with jax.transfer_guard("disallow"):
            value, coil_gradient, inner_gradient = value_and_gradients(
                coil_dofs,
                inner_dofs,
                True,
                True,
            )
            jax.block_until_ready((value, coil_gradient, inner_gradient))

    np.testing.assert_allclose(np.asarray(value), -3.25)
    np.testing.assert_array_equal(np.asarray(coil_gradient), np.ones(2))
    np.testing.assert_array_equal(np.asarray(inner_gradient), np.ones(6))
