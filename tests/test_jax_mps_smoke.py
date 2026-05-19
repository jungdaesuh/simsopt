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

    if not any(device.platform.lower() == "mps" for device in jax.devices()):
        pytest.skip("jax_plugins.mps importable but no MPS device available")

    import simsopt.backend as backend

    backend.set_backend("jax_mps_smoke")

    import jax.numpy as jnp

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

    b = np.asarray(
        biot_savart_B(
            jnp.asarray(points),
            jnp.asarray(gamma[None, :, :]),
            jnp.asarray(gammadash[None, :, :]),
            jnp.asarray([current]),
        )
    )

    assert b.shape == (points.shape[0], 3)
    assert np.all(np.isfinite(b))

    expected_bz_center = MU0 * current / (2 * radius)
    expected_bz_offaxis = MU0 * current * radius**2 / (2 * (radius**2 + 0.5**2) ** 1.5)
    assert b[0, 2] == pytest.approx(expected_bz_center, rel=_SMOKE_RTOL)
    assert b[1, 2] == pytest.approx(expected_bz_offaxis, rel=_SMOKE_RTOL)
