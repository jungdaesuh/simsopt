from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from simsopt.jax_core.mps_custom_call_smoke import (
    SIMSOPT_MPS_CUSTOM_CALL_SMOKE_TARGET,
    simsopt_mps_custom_call_smoke,
)


def test_simsopt_mps_custom_call_smoke_lowers_to_named_stablehlo_target():
    x = jnp.arange(4, dtype=jnp.float32)

    lowered = jax.jit(simsopt_mps_custom_call_smoke).lower(x).as_text()

    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_CUSTOM_CALL_SMOKE_TARGET}" in lowered


@pytest.mark.mps
def test_simsopt_mps_custom_call_smoke_executes_on_real_mps_backend():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    x = jax.device_put(host_x, mps_devices[0])

    y = jax.jit(simsopt_mps_custom_call_smoke)(x)
    y.block_until_ready()

    np.testing.assert_allclose(np.asarray(y), np.asarray(x))
    assert y.device.platform.lower() == "mps"
