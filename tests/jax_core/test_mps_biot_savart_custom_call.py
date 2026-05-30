from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from simsopt.jax_core.mps_biot_savart_custom_call import (
    SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET,
    simsopt_mps_biot_savart_b_group,
)


def _inputs(dtype=jnp.float32):
    points = jnp.asarray(
        [[0.2, -0.1, 0.3], [0.5, 0.4, -0.2]],
        dtype=dtype,
    )
    gammas = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=dtype,
    )
    gammadashs = jnp.asarray(
        [
            [[0.0, 1.0, 0.25], [0.0, 0.5, 0.75], [0.0, 0.25, 1.0]],
            [[0.5, 0.0, 1.0], [0.25, 0.0, 0.5], [0.75, 1.0, 0.0]],
        ],
        dtype=dtype,
    )
    currents = jnp.asarray([1.5, -0.75], dtype=dtype)
    return points, gammas, gammadashs, currents


def _biot_savart_b_oracle(points, gammas, gammadashs, currents):
    host_points = np.asarray(points, dtype=np.float32)
    host_gammas = np.asarray(gammas, dtype=np.float32)
    host_gammadashs = np.asarray(gammadashs, dtype=np.float32)
    host_currents = np.asarray(currents, dtype=np.float32)
    diff = host_gammas[None, :, :, :] - host_points[:, None, None, :]
    radius_squared = np.sum(diff * diff, axis=-1)
    cross = np.cross(diff, host_gammadashs[None, :, :, :])
    weighted = cross * (1.0 / np.sqrt(radius_squared) / radius_squared)[..., None]
    weighted = weighted * host_currents[None, :, None, None]
    return np.float32(1.0e-7) * np.sum(weighted, axis=(1, 2)) / host_gammas.shape[1]


def test_simsopt_mps_biot_savart_b_group_lowers_to_named_stablehlo_target():
    inputs = _inputs()

    lowered = jax.jit(simsopt_mps_biot_savart_b_group).lower(*inputs).as_text()

    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET}" in lowered


def test_simsopt_mps_biot_savart_b_group_rejects_invalid_shapes_and_dtypes():
    points, gammas, gammadashs, currents = _inputs()

    with pytest.raises(ValueError, match=r"points must have shape"):
        simsopt_mps_biot_savart_b_group(
            points.reshape(2, 3, 1), gammas, gammadashs, currents
        )

    with pytest.raises(ValueError, match=r"gammas must have shape"):
        simsopt_mps_biot_savart_b_group(
            points, gammas.reshape(2, 9), gammadashs, currents
        )

    with pytest.raises(ValueError, match=r"at least one quadrature point"):
        simsopt_mps_biot_savart_b_group(
            points,
            gammas[:, :0, :],
            gammadashs[:, :0, :],
            currents,
        )

    with pytest.raises(ValueError, match=r"gammadashs must have the same shape"):
        simsopt_mps_biot_savart_b_group(points, gammas, gammadashs[:, :2, :], currents)

    with pytest.raises(ValueError, match=r"currents must have shape"):
        simsopt_mps_biot_savart_b_group(points, gammas, gammadashs, currents[:1])

    with pytest.raises(ValueError, match=r"currents must have the same dtype"):
        simsopt_mps_biot_savart_b_group(
            points,
            gammas,
            gammadashs,
            currents.astype(jnp.float16),
        )

    with pytest.raises(ValueError, match=r"must be float32"):
        simsopt_mps_biot_savart_b_group(
            points.astype(jnp.float16),
            gammas.astype(jnp.float16),
            gammadashs.astype(jnp.float16),
            currents.astype(jnp.float16),
        )


@pytest.mark.mps
def test_simsopt_mps_biot_savart_b_group_matches_oracle_on_real_mps_backend():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _inputs()
    mps_inputs = tuple(jax.device_put(value, mps_devices[0]) for value in host_inputs)

    actual = jax.jit(simsopt_mps_biot_savart_b_group)(*mps_inputs)
    actual.block_until_ready()
    expected = _biot_savart_b_oracle(*host_inputs)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-12
    )
    assert actual.device.platform.lower() == "mps"
