"""Smoke test for the jax_mps_smoke lane.

Skipped unless ``jax_plugins.mps`` is importable. Runs only in the parallel
``envs/jax-mps.yml`` env on Apple Silicon. The smoke tier makes no parity
claim against the C++ or CPU JAX oracles; this test asserts only that the
Apple-GPU lane produces finite output of the correct shape and order of
magnitude for a circular-loop B-field. MLX is float32-only, so the
magnitude check uses a 1% relative tolerance against the closed-form
axial B-field.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax_plugins.mps")

pytestmark = pytest.mark.mps

MU0 = 4.0 * np.pi * 1e-7
_SMOKE_RTOL = 1e-2


def test_biot_savart_b_smoke_runs_on_mps():
    """``biot_savart_B`` runs on the mps lane and matches the closed-form B-field
    of a circular loop on its axis to within smoke tolerance."""
    import jax

    mps_devices = [device for device in jax.devices() if device.platform.lower() == "mps"]
    if not mps_devices:
        pytest.skip("jax_plugins.mps importable but no MPS device available")
    mps_device = mps_devices[0]

    import simsopt.backend as backend

    backend.set_backend("jax_mps_smoke")

    from simsopt.jax_core.biotsavart import biot_savart_B

    nquad = 64
    radius = 1.0
    current = 5.0e4
    phi = np.linspace(0.0, 2 * np.pi, nquad, endpoint=False)
    gamma = np.stack(
        [radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)], axis=-1
    )
    gammadash = np.stack(
        [
            -radius * np.sin(phi) * 2 * np.pi,
            radius * np.cos(phi) * 2 * np.pi,
            np.zeros_like(phi),
        ],
        axis=-1,
    )
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

    points_device = jax.device_put(np.asarray(points, dtype=np.float32), mps_device)
    gamma_device = jax.device_put(
        np.asarray(gamma[None, :, :], dtype=np.float32), mps_device
    )
    gammadash_device = jax.device_put(
        np.asarray(gammadash[None, :, :], dtype=np.float32), mps_device
    )
    currents_device = jax.device_put(
        np.asarray([current], dtype=np.float32), mps_device
    )

    b_device = biot_savart_B(
        points_device,
        gamma_device,
        gammadash_device,
        currents_device,
    )
    b_device.block_until_ready()
    with jax.transfer_guard_device_to_host("allow"):
        b = np.asarray(jax.device_get(b_device))

    assert b.shape == (points.shape[0], 3)
    assert np.all(np.isfinite(b))

    expected_bz_center = MU0 * current / (2 * radius)
    expected_bz_offaxis = MU0 * current * radius**2 / (2 * (radius**2 + 0.5**2) ** 1.5)
    assert b[0, 2] == pytest.approx(expected_bz_center, rel=_SMOKE_RTOL)
    assert b[1, 2] == pytest.approx(expected_bz_offaxis, rel=_SMOKE_RTOL)
