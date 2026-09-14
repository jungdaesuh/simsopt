"""Checks for the fixed-data analytic Boozer residual Jacobian kernel."""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import simsoptpp

from simsopt_jax.geo.boozer_residual import (
    boozer_residual_vector_and_jacobian,
)


@pytest.fixture
def analytic_inputs():
    """Return a nonsymmetric derivative-direction-first synthetic state."""
    rng = np.random.default_rng(20260908)
    nphi, ntheta, ns = 3, 4, 5
    grid_shape = (nphi, ntheta)
    B = rng.normal(size=grid_shape + (3,)) + np.array([0.25, -0.4, 1.5])
    dB_dX = rng.normal(scale=0.3, size=grid_shape + (3, 3))
    dB_dX[..., 0, 1] += 0.7
    dB_dX[..., 1, 0] -= 0.2
    assert not np.allclose(dB_dX, np.swapaxes(dB_dX, -1, -2))
    xphi = rng.normal(scale=0.4, size=grid_shape + (3,))
    xtheta = rng.normal(scale=0.4, size=grid_shape + (3,))
    dx_ds = rng.normal(scale=0.2, size=grid_shape + (3, ns))
    dxphi_ds = rng.normal(scale=0.2, size=grid_shape + (3, ns))
    dxtheta_ds = rng.normal(scale=0.2, size=grid_shape + (3, ns))
    return {
        "G": 1.25,
        "iota": -0.37,
        "B": jnp.asarray(B),
        "dB_dX": jnp.asarray(dB_dX),
        "xphi": jnp.asarray(xphi),
        "xtheta": jnp.asarray(xtheta),
        "dx_ds": jnp.asarray(dx_ds),
        "dxphi_ds": jnp.asarray(dxphi_ds),
        "dxtheta_ds": jnp.asarray(dxtheta_ds),
        "ns": ns,
    }


def _state_residual(state, inputs, *, weight_inv_modB):
    """Compose the supplied first-order field/surface model independently."""
    ns = inputs["ns"]
    surface_dofs = state[:ns]
    iota = state[ns]
    G = state[ns + 1]
    B = inputs["B"] + jnp.einsum(
        "...km,...ka,a->...m",
        inputs["dB_dX"],
        inputs["dx_ds"],
        surface_dofs,
    )
    xphi = inputs["xphi"] + jnp.einsum(
        "...ma,a->...m", inputs["dxphi_ds"], surface_dofs
    )
    xtheta = inputs["xtheta"] + jnp.einsum(
        "...ma,a->...m", inputs["dxtheta_ds"], surface_dofs
    )
    B2 = jnp.sum(B * B, axis=-1)
    residual = G * B - B2[..., None] * (xphi + iota * xtheta)
    if weight_inv_modB:
        residual = residual / jnp.sqrt(B2)[..., None]
    return residual.reshape((-1,))


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.parametrize("optimize_G", [False, True])
def test_matches_independent_composed_ad(
    analytic_inputs, *, optimize_G, weight_inv_modB
):
    """Both weighting modes and G-column modes match an independent AD model."""
    inputs = analytic_inputs
    args = (
        inputs["G"],
        inputs["iota"],
        inputs["B"],
        inputs["dB_dX"],
        inputs["xphi"],
        inputs["xtheta"],
        inputs["dx_ds"],
        inputs["dxphi_ds"],
        inputs["dxtheta_ds"],
    )
    residual, jacobian = boozer_residual_vector_and_jacobian(
        *args,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )

    state = jnp.concatenate(
        (
            jnp.zeros((inputs["ns"],), dtype=inputs["B"].dtype),
            jnp.asarray([inputs["iota"], inputs["G"]], dtype=inputs["B"].dtype),
        )
    )
    independent_residual = _state_residual(
        state, inputs, weight_inv_modB=weight_inv_modB
    )
    independent_jacobian = jax.jacobian(
        lambda value: _state_residual(
            value, inputs, weight_inv_modB=weight_inv_modB
        )
    )(state)
    expected_columns = inputs["ns"] + 1 + int(optimize_G)

    assert residual.shape == (3 * 3 * 4,)
    assert jacobian.shape == (3 * 3 * 4, expected_columns)
    np.testing.assert_allclose(residual, independent_residual, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        jacobian,
        independent_jacobian[:, :expected_columns],
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("optimize_G", [False, True])
@pytest.mark.parametrize("weight_inv_modB", [False, True])
def test_surface_columns_match_native_dresidual_dc(
    analytic_inputs, *, optimize_G, weight_inv_modB
):
    """Surface/iota/G columns match native derivatives with either weight."""
    inputs = analytic_inputs
    B = np.asarray(inputs["B"])
    dB_dX = np.asarray(inputs["dB_dX"])
    xphi = np.asarray(inputs["xphi"])
    xtheta = np.asarray(inputs["xtheta"])
    dx_ds = np.asarray(inputs["dx_ds"])
    dxphi_ds = np.asarray(inputs["dxphi_ds"])
    dxtheta_ds = np.asarray(inputs["dxtheta_ds"])
    iota = float(inputs["iota"])
    G = float(inputs["G"])
    tang = xphi + iota * xtheta
    B2 = np.sum(B * B, axis=-1)
    dB_ds = np.einsum("...km,...ka->...ma", dB_dX, dx_ds)
    native_surface = simsoptpp.boozer_dresidual_dc(
        G,
        dB_ds,
        B,
        tang,
        B2,
        dxphi_ds,
        iota,
        dxtheta_ds,
    )
    if weight_inv_modB:
        weight = 1.0 / np.sqrt(B2)
        residual = G * B - B2[..., None] * tang
        dB2_ds = 2.0 * np.einsum("...m,...ma->...a", B, dB_ds)
        dweight_ds = -0.5 * dB2_ds / B2[..., None] ** 1.5
        surface_columns = (
            weight[..., None, None] * native_surface
            + residual[..., :, None] * dweight_ds[..., None, :]
        )
        iota_column = weight[..., None] * (-B2[..., None] * xtheta)
        g_column = weight[..., None] * B
    else:
        surface_columns = native_surface
        iota_column = -B2[..., None] * xtheta
        g_column = B
    expected = np.concatenate(
        (
            surface_columns,
            iota_column[..., None],
            g_column[..., None],
        ),
        axis=-1,
    ).reshape((-1, inputs["ns"] + 2))
    if not optimize_G:
        expected = expected[:, :-1]

    _, jacobian = boozer_residual_vector_and_jacobian(
        inputs["G"],
        inputs["iota"],
        inputs["B"],
        inputs["dB_dX"],
        inputs["xphi"],
        inputs["xtheta"],
        inputs["dx_ds"],
        inputs["dxphi_ds"],
        inputs["dxtheta_ds"],
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    np.testing.assert_allclose(jacobian, expected, rtol=2e-11, atol=2e-11)
