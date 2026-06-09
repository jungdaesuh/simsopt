"""Regression tests for small JAX kernel numeric helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from simsopt_jax.core._spline_utils import _safe_divide
from simsopt_jax.core.biotsavart import (
    _radius_squared,
    biot_savart_A,
    biot_savart_B,
    biot_savart_dB_by_dX,
)
from simsopt_jax.core.curve_xyz_fourier import _constant_row


def test_radius_squared_preserves_zero_singularity():
    diff = jnp.zeros((1, 3), dtype=jnp.float64)

    radius_squared = _radius_squared(diff)

    np.testing.assert_allclose(np.asarray(radius_squared), np.zeros((1,)))


def test_biotsavart_point_singularity_gradient_is_nonfinite():
    def singular_kernel(x):
        diff = jnp.reshape(x, (1, 3))
        r2 = _radius_squared(diff)[0]
        return x[0] / (r2**1.5)

    gradient = jax.grad(singular_kernel)(jnp.zeros((3,), dtype=jnp.float64))

    assert not np.all(np.isfinite(np.asarray(gradient)))


def test_biot_savart_public_kernels_preserve_point_singularity():
    points = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64)
    gammas = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float64)
    gammadashs = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64)
    currents = jnp.asarray([1.0], dtype=jnp.float64)

    for kernel in (biot_savart_A, biot_savart_B, biot_savart_dB_by_dX):
        value = kernel(points, gammas, gammadashs, currents)

        assert not np.all(np.isfinite(np.asarray(value)))


def test_safe_divide_has_finite_zero_denominator_gradient():
    def scalar_value(denominator):
        return _safe_divide(jnp.asarray(2.0, dtype=jnp.float64), denominator)

    gradient = jax.grad(scalar_value)(jnp.asarray(0.0, dtype=jnp.float64))

    assert np.isfinite(np.asarray(gradient))
    np.testing.assert_allclose(np.asarray(gradient), 0.0)


def test_constant_row_accepts_traced_value():
    @jax.jit
    def row_for(value):
        return _constant_row(3, value, reference=jnp.asarray(1.0, dtype=jnp.float64))

    np.testing.assert_allclose(
        np.asarray(row_for(jnp.asarray(1.0, dtype=jnp.float64))),
        np.ones((1, 3)),
    )
    np.testing.assert_allclose(
        np.asarray(row_for(jnp.asarray(0.0, dtype=jnp.float64))),
        np.zeros((1, 3)),
    )
