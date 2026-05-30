from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from simsopt.jax_core.mps_boozer_residual_custom_call import (
    SIMSOPT_MPS_BOOZER_RESIDUAL_VECTOR_TARGET,
    SIMSOPT_MPS_BOOZER_RESIDUAL_VALUE_GRAD_TARGET,
    SIMSOPT_MPS_BOOZER_RESIDUAL_VJP_TARGET,
    SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VECTOR_TARGET,
    SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VALUE_GRAD_TARGET,
    SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VJP_TARGET,
    simsopt_mps_boozer_residual_vector,
    simsopt_mps_boozer_residual_value_and_grad,
    simsopt_mps_boozer_residual_vector_with_vjp,
    simsopt_mps_boozer_residual_vjp,
)
from simsopt.geo.boozer_residual_jax import (
    boozer_residual_vector as reference_boozer_residual_vector,
)


def _restore_backend_config(config) -> None:
    from simsopt.backend import set_backend

    set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        disable_jit=config.disable_jit,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        xla_gpu_preallocate=config.xla_gpu_preallocate,
        xla_gpu_mem_fraction=config.xla_gpu_mem_fraction,
        xla_gpu_allocator=config.xla_gpu_allocator,
        tf_gpu_allocator=config.tf_gpu_allocator,
        configure_runtime=False,
    )


@contextmanager
def _temporary_backend(mode: str):
    from simsopt.backend import get_backend_config, set_backend

    previous = get_backend_config()
    try:
        set_backend(mode, configure_runtime=False)
        yield
    finally:
        _restore_backend_config(previous)


def _residual_inputs(dtype=jnp.float32):
    G = jnp.asarray(1.7, dtype=dtype)
    iota = jnp.asarray(0.31, dtype=dtype)
    B = jnp.asarray(
        [
            [[1.0, 0.2, -0.4], [0.3, 0.9, 0.5]],
            [[-0.7, 0.4, 1.2], [0.8, -1.1, 0.6]],
        ],
        dtype=dtype,
    )
    xphi = jnp.asarray(np.linspace(-0.3, 0.6, 12).reshape(2, 2, 3), dtype=dtype)
    xtheta = jnp.asarray(np.linspace(0.5, -0.2, 12).reshape(2, 2, 3), dtype=dtype)
    return G, iota, B, xphi, xtheta


def _residual_cotangent(dtype=jnp.float32):
    return jnp.asarray(np.linspace(0.2, 1.3, 12), dtype=dtype)


def _normalized_scalar_from_vector(residual_vector):
    return jnp.asarray(0.5, dtype=residual_vector.dtype) * (
        jnp.sum(residual_vector * residual_vector) / residual_vector.size
    )


def _reference_residual_scalar(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    return _normalized_scalar_from_vector(
        reference_boozer_residual_vector(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )
    )


def _custom_residual_scalar(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    return _normalized_scalar_from_vector(
        simsopt_mps_boozer_residual_vector_with_vjp(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )
    )


def _boozer_residual_vector_oracle(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    host_G = np.asarray(G, dtype=np.float32)
    host_iota = np.asarray(iota, dtype=np.float32)
    host_B = np.asarray(B, dtype=np.float32)
    host_xphi = np.asarray(xphi, dtype=np.float32)
    host_xtheta = np.asarray(xtheta, dtype=np.float32)
    tangent = host_xphi + host_iota * host_xtheta
    B2 = np.sum(host_B * host_B, axis=-1)
    residual = host_G * host_B - B2[..., None] * tangent
    if weight_inv_modB:
        residual = (1.0 / np.sqrt(B2))[..., None] * residual
    return residual.reshape(-1)


def _boozer_residual_vjp_oracle(
    G,
    iota,
    B,
    xphi,
    xtheta,
    residual_cotangent,
    *,
    weight_inv_modB,
):
    host_G = np.asarray(G, dtype=np.float32)
    host_iota = np.asarray(iota, dtype=np.float32)
    host_B = np.asarray(B, dtype=np.float32)
    host_xphi = np.asarray(xphi, dtype=np.float32)
    host_xtheta = np.asarray(xtheta, dtype=np.float32)
    cotangent = np.asarray(residual_cotangent, dtype=np.float32).reshape(host_B.shape)
    tangent = host_xphi + host_iota * host_xtheta
    B2 = np.sum(host_B * host_B, axis=-1)
    numerator = host_G * host_B - B2[..., None] * tangent

    if weight_inv_modB:
        inv_modB = 1.0 / np.sqrt(B2)
        grad_B = inv_modB[..., None] * (
            host_G * cotangent
            - 2.0 * np.sum(cotangent * tangent, axis=-1)[..., None] * host_B
        )
        grad_B -= (
            np.sum(cotangent * numerator, axis=-1) * inv_modB * inv_modB * inv_modB
        )[..., None] * host_B
        weighted_cotangent = inv_modB[..., None] * cotangent
        grad_xphi = -weighted_cotangent * B2[..., None]
        grad_xtheta = -host_iota * weighted_cotangent * B2[..., None]
        grad_G = np.sum(weighted_cotangent * host_B)
        grad_iota = np.sum(weighted_cotangent * (-B2[..., None] * host_xtheta))
    else:
        grad_B = (
            host_G * cotangent
            - 2.0 * np.sum(cotangent * tangent, axis=-1)[..., None] * host_B
        )
        grad_xphi = -B2[..., None] * cotangent
        grad_xtheta = -host_iota * B2[..., None] * cotangent
        grad_G = np.sum(cotangent * host_B)
        grad_iota = np.sum(cotangent * (-B2[..., None] * host_xtheta))
    return (
        np.asarray(grad_G, dtype=np.float32),
        np.asarray(grad_iota, dtype=np.float32),
        grad_B,
        grad_xphi,
        grad_xtheta,
    )


def _boozer_residual_value_grad_oracle(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    residual = _boozer_residual_vector_oracle(
        G,
        iota,
        B,
        xphi,
        xtheta,
        weight_inv_modB=weight_inv_modB,
    )
    value = np.float32(0.5) * np.sum(residual * residual) / np.float32(residual.size)
    cotangent = residual / np.float32(residual.size)
    gradient = _boozer_residual_vjp_oracle(
        G,
        iota,
        B,
        xphi,
        xtheta,
        cotangent,
        weight_inv_modB=weight_inv_modB,
    )
    return np.asarray(value, dtype=np.float32), gradient


@pytest.mark.parametrize(
    ("weight_inv_modB", "target"),
    [
        (False, SIMSOPT_MPS_BOOZER_RESIDUAL_VECTOR_TARGET),
        (True, SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VECTOR_TARGET),
    ],
)
def test_simsopt_mps_boozer_residual_vector_lowers_to_named_stablehlo_target(
    weight_inv_modB,
    target,
):
    args = _residual_inputs()

    lowered = (
        jax.jit(
            lambda G, iota, B, xphi, xtheta: simsopt_mps_boozer_residual_vector(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=weight_inv_modB,
            )
        )
        .lower(*args)
        .as_text()
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{target}" in lowered


@pytest.mark.parametrize(
    ("weight_inv_modB", "target"),
    [
        (False, SIMSOPT_MPS_BOOZER_RESIDUAL_VJP_TARGET),
        (True, SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VJP_TARGET),
    ],
)
def test_simsopt_mps_boozer_residual_vjp_lowers_to_named_stablehlo_target(
    weight_inv_modB,
    target,
):
    args = _residual_inputs()
    cotangent = _residual_cotangent()

    lowered = (
        jax.jit(
            lambda G, iota, B, xphi, xtheta, residual_cotangent: (
                simsopt_mps_boozer_residual_vjp(
                    G,
                    iota,
                    B,
                    xphi,
                    xtheta,
                    residual_cotangent,
                    weight_inv_modB=weight_inv_modB,
                )
            )
        )
        .lower(*args, cotangent)
        .as_text()
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{target}" in lowered


@pytest.mark.parametrize(
    ("weight_inv_modB", "target"),
    [
        (False, SIMSOPT_MPS_BOOZER_RESIDUAL_VALUE_GRAD_TARGET),
        (True, SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VALUE_GRAD_TARGET),
    ],
)
def test_simsopt_mps_boozer_residual_value_grad_lowers_to_named_stablehlo_target(
    weight_inv_modB,
    target,
):
    args = _residual_inputs()

    lowered = (
        jax.jit(
            lambda G, iota, B, xphi, xtheta: simsopt_mps_boozer_residual_value_and_grad(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=weight_inv_modB,
            )
        )
        .lower(*args)
        .as_text()
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{target}" in lowered


@pytest.mark.parametrize(
    ("weight_inv_modB", "vector_target", "vjp_target"),
    [
        (
            False,
            SIMSOPT_MPS_BOOZER_RESIDUAL_VECTOR_TARGET,
            SIMSOPT_MPS_BOOZER_RESIDUAL_VJP_TARGET,
        ),
        (
            True,
            SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VECTOR_TARGET,
            SIMSOPT_MPS_BOOZER_WEIGHTED_RESIDUAL_VJP_TARGET,
        ),
    ],
)
def test_simsopt_mps_boozer_residual_value_and_grad_lowers_to_paired_targets(
    weight_inv_modB,
    vector_target,
    vjp_target,
):
    args = _residual_inputs()

    lowered = (
        jax.jit(
            jax.value_and_grad(
                lambda G, iota, B, xphi, xtheta: _custom_residual_scalar(
                    G,
                    iota,
                    B,
                    xphi,
                    xtheta,
                    weight_inv_modB=weight_inv_modB,
                ),
                argnums=(0, 1, 2, 3, 4),
            )
        )
        .lower(*args)
        .as_text()
    )

    assert "stablehlo.custom_call" in lowered
    assert f"@{vector_target}" in lowered
    assert f"@{vjp_target}" in lowered


def test_simsopt_mps_boozer_residual_vector_rejects_invalid_inputs():
    G, iota, B, xphi, xtheta = _residual_inputs()

    with pytest.raises(ValueError, match=r"G must be a scalar"):
        simsopt_mps_boozer_residual_vector(jnp.reshape(G, (1,)), iota, B, xphi, xtheta)

    with pytest.raises(ValueError, match=r"iota must be a scalar"):
        simsopt_mps_boozer_residual_vector(G, jnp.reshape(iota, (1,)), B, xphi, xtheta)

    with pytest.raises(ValueError, match=r"B must have shape"):
        simsopt_mps_boozer_residual_vector(G, iota, B.reshape(2, 6), xphi, xtheta)

    with pytest.raises(ValueError, match=r"at least one surface grid point"):
        simsopt_mps_boozer_residual_vector(G, iota, B[:0], xphi[:0], xtheta[:0])

    with pytest.raises(ValueError, match=r"xphi must have the same shape"):
        simsopt_mps_boozer_residual_vector(G, iota, B, xphi[:, :1, :], xtheta)

    with pytest.raises(ValueError, match=r"xtheta must have the same shape"):
        simsopt_mps_boozer_residual_vector(G, iota, B, xphi, xtheta[:, :1, :])

    with pytest.raises(ValueError, match=r"G, iota, and B must share a dtype"):
        simsopt_mps_boozer_residual_vector(G.astype(jnp.float16), iota, B, xphi, xtheta)

    with pytest.raises(ValueError, match=r"B, xphi, and xtheta must share a dtype"):
        simsopt_mps_boozer_residual_vector(G, iota, B, xphi.astype(jnp.float16), xtheta)

    with pytest.raises(ValueError, match=r"must be float32"):
        simsopt_mps_boozer_residual_vector(
            G.astype(jnp.float16),
            iota.astype(jnp.float16),
            B.astype(jnp.float16),
            xphi.astype(jnp.float16),
            xtheta.astype(jnp.float16),
        )


def test_simsopt_mps_boozer_residual_vjp_rejects_invalid_cotangent():
    G, iota, B, xphi, xtheta = _residual_inputs()
    cotangent = _residual_cotangent()

    with pytest.raises(ValueError, match=r"residual_cotangent must have shape"):
        simsopt_mps_boozer_residual_vjp(G, iota, B, xphi, xtheta, cotangent[:-1])

    with pytest.raises(
        ValueError, match=r"residual_cotangent must have the same dtype"
    ):
        simsopt_mps_boozer_residual_vjp(
            G,
            iota,
            B,
            xphi,
            xtheta,
            cotangent.astype(jnp.float16),
        )


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.mps
def test_simsopt_mps_boozer_residual_vector_matches_oracle_on_real_mps_backend(
    weight_inv_modB,
):
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _residual_inputs()
    mps_inputs = tuple(jax.device_put(value, mps_devices[0]) for value in host_inputs)

    actual = jax.jit(
        lambda G, iota, B, xphi, xtheta: simsopt_mps_boozer_residual_vector(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )
    )(*mps_inputs)
    actual.block_until_ready()
    expected = _boozer_residual_vector_oracle(
        *host_inputs,
        weight_inv_modB=weight_inv_modB,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual.device.platform.lower() == "mps"


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.mps
def test_simsopt_mps_boozer_residual_vjp_matches_oracle_on_real_mps_backend(
    weight_inv_modB,
):
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _residual_inputs()
    host_cotangent = _residual_cotangent()
    mps_inputs = tuple(jax.device_put(value, mps_devices[0]) for value in host_inputs)
    mps_cotangent = jax.device_put(host_cotangent, mps_devices[0])

    actual = jax.jit(
        lambda G, iota, B, xphi, xtheta, residual_cotangent: (
            simsopt_mps_boozer_residual_vjp(
                G,
                iota,
                B,
                xphi,
                xtheta,
                residual_cotangent,
                weight_inv_modB=weight_inv_modB,
            )
        )
    )(*mps_inputs, mps_cotangent)
    jax.block_until_ready(actual)
    expected = _boozer_residual_vjp_oracle(
        *host_inputs,
        host_cotangent,
        weight_inv_modB=weight_inv_modB,
    )

    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            expected_leaf,
            rtol=1e-5,
            atol=1e-6,
        )
        assert actual_leaf.device.platform.lower() == "mps"


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.mps
def test_simsopt_mps_boozer_residual_value_grad_matches_oracle_on_real_mps_backend(
    weight_inv_modB,
):
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _residual_inputs()
    mps_inputs = tuple(jax.device_put(value, mps_devices[0]) for value in host_inputs)

    actual_value, actual_grad = jax.jit(
        lambda G, iota, B, xphi, xtheta: simsopt_mps_boozer_residual_value_and_grad(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        )
    )(*mps_inputs)
    jax.block_until_ready((actual_value, actual_grad))
    expected_value, expected_grad = _boozer_residual_value_grad_oracle(
        *host_inputs,
        weight_inv_modB=weight_inv_modB,
    )

    np.testing.assert_allclose(
        np.asarray(actual_value),
        expected_value,
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual_value.device.platform.lower() == "mps"
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            expected_leaf,
            rtol=1e-5,
            atol=1e-6,
        )
        assert actual_leaf.device.platform.lower() == "mps"


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.mps
def test_simsopt_mps_boozer_residual_value_and_grad_matches_current_jax_path(
    weight_inv_modB,
):
    """MPS-smoke gate, not CPU-oracle parity: compare against JAX residuals on MPS."""
    mps_devices = tuple(
        device for device in jax.devices() if device.platform.lower() == "mps"
    )
    if not mps_devices:
        pytest.skip("requires a JAX MPS device")

    host_inputs = _residual_inputs()
    mps_inputs = tuple(jax.device_put(value, mps_devices[0]) for value in host_inputs)

    reference_value_and_grad = jax.value_and_grad(
        lambda G, iota, B, xphi, xtheta: _reference_residual_scalar(
            G,
            iota,
            B,
            xphi,
            xtheta,
            weight_inv_modB=weight_inv_modB,
        ),
        argnums=(0, 1, 2, 3, 4),
    )
    custom_value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda G, iota, B, xphi, xtheta: _custom_residual_scalar(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=weight_inv_modB,
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )

    with _temporary_backend("jax_mps_smoke"):
        expected_value, expected_grad = jax.jit(reference_value_and_grad)(*mps_inputs)
        actual_value, actual_grad = custom_value_and_grad(*mps_inputs)
    jax.block_until_ready((actual_value, actual_grad))

    np.testing.assert_allclose(
        np.asarray(actual_value),
        np.asarray(expected_value),
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual_value.device.platform.lower() == "mps"
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=1e-5,
            atol=1e-6,
        )
        assert actual_leaf.device.platform.lower() == "mps"
