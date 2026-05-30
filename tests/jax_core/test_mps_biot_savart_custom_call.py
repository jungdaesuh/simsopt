from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from simsopt.jax_core.mps_biot_savart_custom_call import (
    SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET,
    SIMSOPT_MPS_BIOT_SAVART_B_VJP_GROUP_TARGET,
    simsopt_mps_biot_savart_b_group,
    simsopt_mps_biot_savart_b_vjp_group,
)
from simsopt.jax_core.biotsavart import biot_savart_B


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


def _field_cotangent(dtype=jnp.float32):
    return jnp.asarray(
        [[0.75, -0.25, 0.5], [-0.2, 0.1, 0.3]],
        dtype=dtype,
    )


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


def _biot_savart_b_vjp_jax_oracle(
    points,
    field_cotangent,
    gammas,
    gammadashs,
    currents,
):
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        cpu_points = jax.device_put(points, cpu_device)
        cpu_field_cotangent = jax.device_put(field_cotangent, cpu_device)
        cpu_gammas = jax.device_put(gammas, cpu_device)
        cpu_gammadashs = jax.device_put(gammadashs, cpu_device)
        cpu_currents = jax.device_put(currents, cpu_device)

        def forward(group_gammas, group_gammadashs, group_currents):
            return biot_savart_B(
                cpu_points,
                group_gammas,
                group_gammadashs,
                group_currents,
            )

        field_value, pullback = jax.vjp(
            forward, cpu_gammas, cpu_gammadashs, cpu_currents
        )
        cpu_field_cotangent = cpu_field_cotangent.astype(field_value.dtype)
        return tuple(
            np.asarray(leaf, dtype=np.float32) for leaf in pullback(cpu_field_cotangent)
        )


def test_simsopt_mps_biot_savart_b_group_lowers_to_named_stablehlo_target():
    inputs = _inputs()

    lowered = jax.jit(simsopt_mps_biot_savart_b_group).lower(*inputs).as_text()

    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET}" in lowered


def test_simsopt_mps_biot_savart_b_vjp_group_lowers_to_named_stablehlo_target():
    points, gammas, gammadashs, currents = _inputs()
    field_cotangent = _field_cotangent()

    lowered = (
        jax.jit(simsopt_mps_biot_savart_b_vjp_group)
        .lower(points, field_cotangent, gammas, gammadashs, currents)
        .as_text()
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{SIMSOPT_MPS_BIOT_SAVART_B_VJP_GROUP_TARGET}" in lowered


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


def test_simsopt_mps_biot_savart_b_vjp_group_rejects_invalid_cotangent():
    points, gammas, gammadashs, currents = _inputs()
    field_cotangent = _field_cotangent()

    with pytest.raises(ValueError, match=r"field_cotangent must have the same shape"):
        simsopt_mps_biot_savart_b_vjp_group(
            points,
            field_cotangent[:1],
            gammas,
            gammadashs,
            currents,
        )

    with pytest.raises(ValueError, match=r"field_cotangent must have the same dtype"):
        simsopt_mps_biot_savart_b_vjp_group(
            points,
            field_cotangent.astype(jnp.float16),
            gammas,
            gammadashs,
            currents,
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


@pytest.mark.mps
def test_simsopt_mps_biot_savart_b_vjp_group_matches_jax_vjp_on_real_mps_backend():
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _inputs()
    host_field_cotangent = _field_cotangent()
    mps_inputs = (
        jax.device_put(host_inputs[0], mps_devices[0]),
        jax.device_put(host_field_cotangent, mps_devices[0]),
        jax.device_put(host_inputs[1], mps_devices[0]),
        jax.device_put(host_inputs[2], mps_devices[0]),
        jax.device_put(host_inputs[3], mps_devices[0]),
    )

    actual = jax.jit(simsopt_mps_biot_savart_b_vjp_group)(*mps_inputs)
    jax.block_until_ready(actual)
    expected = _biot_savart_b_vjp_jax_oracle(
        host_inputs[0],
        host_field_cotangent,
        host_inputs[1],
        host_inputs[2],
        host_inputs[3],
    )

    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            expected_leaf,
            rtol=1e-5,
            atol=1e-12,
        )
        assert actual_leaf.device.platform.lower() == "mps"
