"""Strict-CUDA coverage for public JAX surface-objective boundaries."""

from __future__ import annotations

from conftest import enable_strict_parity_backend, parity_default_device

import jax
import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax_adapters.geo.surface_objectives import SurfaceSurfaceDistance


def test_surface_surface_distance_stages_public_host_geometry_under_strict_gpu_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    surface1 = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 3, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
    )
    surface1.set_rc(0, 0, 1.0)
    surface1.set_rc(1, 0, 0.12)
    surface1.set_zs(1, 0, 0.12)
    surface2 = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 6, endpoint=False),
    )
    surface2.set_rc(0, 0, 1.05)
    surface2.set_rc(1, 0, 0.14)
    surface2.set_zs(1, 0, 0.14)
    objective = SurfaceSurfaceDistance(
        surface1,
        surface2,
        minimum_distance=0.18,
    )

    with parity_default_device("gpu"):
        with jax.transfer_guard("disallow"):
            value = objective.J()
            derivative = objective.dJ(partials=True)
            gradient1 = derivative(surface1)
            gradient2 = derivative(surface2)

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient1))
    assert np.all(np.isfinite(gradient2))
