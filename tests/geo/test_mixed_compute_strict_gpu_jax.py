"""Strict-CUDA coverage for mixed-precision core geometry staging."""

from __future__ import annotations

from conftest import enable_strict_parity_backend, parity_default_device

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.curve_geometry import _slice_1d_static, _update_1d_static
from simsopt_jax.core.surface_rzfourier import (
    _surface_rz_fourier_derivative_lin_from_spec,
    surface_rz_fourier_spec_from_dofs,
)
from simsopt_jax.geo.surface_fourier import surface_gamma_from_dofs


def test_static_curve_dof_mapping_preserves_mixed_dtype_on_gpu(
    monkeypatch,
    request,
) -> None:
    monkeypatch.setenv("SIMSOPT_PRECISION", "mixed")
    enable_strict_parity_backend(monkeypatch, request, "gpu")

    with parity_default_device("gpu"):
        array = jax.device_put(np.linspace(-2.0, 3.0, num=6, dtype=np.float32))
        replacement = jax.device_put(np.asarray([10.0, 20.0, 30.0], dtype=np.float32))
        with jax.transfer_guard("disallow"):
            segment = _slice_1d_static(
                array,
                2,
                5,
                use_compute_dtype=True,
            )
            updated = _update_1d_static(
                array,
                1,
                replacement,
                use_compute_dtype=True,
            )

    assert segment.dtype == jnp.float32
    assert updated.dtype == jnp.float32


def test_surface_geometry_preserves_mixed_dtype_on_gpu(
    monkeypatch,
    request,
) -> None:
    monkeypatch.setenv("SIMSOPT_PRECISION", "mixed")
    enable_strict_parity_backend(monkeypatch, request, "gpu")

    with parity_default_device("gpu"):
        dofs = jax.device_put(
            np.asarray(
                [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.2, 0.7, -0.4],
                dtype=np.float64,
            )
        )
        quadpoints = jax.device_put(np.asarray([0.0, 0.5], dtype=np.float64))
        with jax.transfer_guard("disallow"):
            gamma = surface_gamma_from_dofs(
                dofs,
                quadpoints,
                quadpoints,
                1,
                0,
                1,
                False,
                None,
                use_compute_dtype=True,
            )

    assert gamma.dtype == jnp.float32


def test_eager_surface_rz_derivative_stages_literals_on_gpu(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")

    with parity_default_device("gpu"):
        dofs = jax.device_put(np.asarray([1.0, 0.1, 0.2], dtype=np.float64))
        quadpoints = jax.device_put(np.asarray([0.0, 0.5], dtype=np.float64))
        spec = surface_rz_fourier_spec_from_dofs(
            dofs,
            quadpoints_phi=quadpoints,
            quadpoints_theta=quadpoints,
            mpol=1,
            ntor=0,
            nfp=1,
            stellsym=True,
        )
        with jax.transfer_guard("disallow"):
            derivative = _surface_rz_fourier_derivative_lin_from_spec(
                spec,
                quadpoints,
                quadpoints,
                1,
                0,
            )
            derivative.block_until_ready()

    assert derivative.dtype == jnp.float64
