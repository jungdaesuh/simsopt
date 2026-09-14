"""CPU tests for intrinsic analytic Boozer least-squares derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp as sopp
from simsopt_jax.geo.boozer_analytic_hessian import (
    boozer_residual_analytic_value_grad,
    boozer_residual_analytic_value_grad_hessian,
)

jax.config.update("jax_enable_x64", True)


def _native_fixture():
    rng = np.random.default_rng(20260908)
    nphi, ntheta, nsurface = 2, 3, 4
    B = rng.normal(size=(nphi, ntheta, 3)) + np.asarray([0.3, -0.2, 1.4])
    d2B = rng.normal(scale=0.13, size=(nphi, ntheta, 3, 3, 3))
    d2B = 0.5 * (d2B + np.swapaxes(d2B, 2, 3))
    return {
        "G": 1.7,
        "iota": -0.31,
        "B": B,
        "dB_dX": rng.normal(scale=0.2, size=(nphi, ntheta, 3, 3)),
        "d2B_dXdX": d2B,
        "xphi": rng.normal(scale=0.4, size=(nphi, ntheta, 3)),
        "xtheta": rng.normal(scale=0.4, size=(nphi, ntheta, 3)),
        "dx_ds": rng.normal(scale=0.3, size=(nphi, ntheta, 3, nsurface)),
        "dxphi_ds": rng.normal(scale=0.3, size=(nphi, ntheta, 3, nsurface)),
        "dxtheta_ds": rng.normal(scale=0.3, size=(nphi, ntheta, 3, nsurface)),
    }


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.parametrize("optimize_G", [False, True])
def test_analytic_derivatives_match_native_nonzero_residual(
    weight_inv_modB,
    optimize_G,
):
    inputs = _native_fixture()
    raw_value, raw_gradient, raw_hessian = sopp.boozer_residual_ds2(
        inputs["G"],
        inputs["iota"],
        inputs["B"],
        inputs["dB_dX"],
        inputs["d2B_dXdX"],
        inputs["xphi"],
        inputs["xtheta"],
        inputs["dx_ds"],
        inputs["dxphi_ds"],
        inputs["dxtheta_ds"],
        weight_inv_modB,
    )
    value, gradient, hessian = boozer_residual_analytic_value_grad_hessian(
        **inputs,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        quadrature_tile_size=2,
    )
    normalization = 3 * inputs["B"].shape[0] * inputs["B"].shape[1]
    selected = slice(None) if optimize_G else slice(0, inputs["dx_ds"].shape[-1] + 1)

    assert float(value) == pytest.approx(raw_value / normalization, rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(raw_gradient)[selected] / normalization,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.asarray(hessian),
        np.asarray(raw_hessian)[selected, selected] / normalization,
        rtol=8e-12,
        atol=8e-12,
    )


def _polynomial_fixture():
    rng = np.random.default_rng(71)
    nphi, ntheta, nsurface = 2, 3, 3
    point_shape = (nphi, ntheta)
    X0 = rng.normal(scale=0.2, size=point_shape + (3,))
    dx_ds = rng.normal(scale=0.3, size=point_shape + (3, nsurface))
    field_offset = rng.normal(size=point_shape + (3,)) + np.asarray([0.1, 0.2, 1.8])
    field_linear = rng.normal(scale=0.4, size=point_shape + (3, 3))
    field_quadratic = rng.normal(scale=0.15, size=point_shape + (3, 3, 3))
    field_quadratic = 0.5 * (
        field_quadratic + np.swapaxes(field_quadratic, -3, -2)
    )
    xphi0 = rng.normal(scale=0.3, size=point_shape + (3,))
    xtheta0 = rng.normal(scale=0.3, size=point_shape + (3,))
    dxphi_ds = rng.normal(scale=0.2, size=point_shape + (3, nsurface))
    dxtheta_ds = rng.normal(scale=0.2, size=point_shape + (3, nsurface))
    surface = rng.normal(scale=0.25, size=(nsurface,))
    iota = -0.27
    G = 1.35

    X = X0 + np.einsum("...ds,s->...d", dx_ds, surface)
    B = (
        field_offset
        + np.einsum("...d,...dk->...k", X, field_linear)
        + 0.5 * np.einsum("...d,...e,...dek->...k", X, X, field_quadratic)
    )
    dB_dX = field_linear + np.einsum("...e,...dek->...dk", X, field_quadratic)
    xphi = xphi0 + np.einsum("...ks,s->...k", dxphi_ds, surface)
    xtheta = xtheta0 + np.einsum("...ks,s->...k", dxtheta_ds, surface)
    return {
        "inputs": {
            "G": G,
            "iota": iota,
            "B": B,
            "dB_dX": dB_dX,
            "d2B_dXdX": field_quadratic,
            "xphi": xphi,
            "xtheta": xtheta,
            "dx_ds": dx_ds,
            "dxphi_ds": dxphi_ds,
            "dxtheta_ds": dxtheta_ds,
        },
        "constants": (
            jnp.asarray(X0),
            jnp.asarray(dx_ds),
            jnp.asarray(field_offset),
            jnp.asarray(field_linear),
            jnp.asarray(field_quadratic),
            jnp.asarray(xphi0),
            jnp.asarray(xtheta0),
            jnp.asarray(dxphi_ds),
            jnp.asarray(dxtheta_ds),
        ),
        "decision": jnp.asarray(np.concatenate((surface, [iota, G]))),
    }


def _polynomial_objective(decision, constants, *, weight_inv_modB, optimize_G):
    (
        X0,
        dx_ds,
        field_offset,
        field_linear,
        field_quadratic,
        xphi0,
        xtheta0,
        dxphi_ds,
        dxtheta_ds,
    ) = constants
    nsurface = dx_ds.shape[-1]
    surface = decision[:nsurface]
    iota = decision[nsurface]
    G = decision[nsurface + 1] if optimize_G else jnp.asarray(1.35)
    X = X0 + jnp.einsum("...ds,s->...d", dx_ds, surface)
    B = (
        field_offset
        + jnp.einsum("...d,...dk->...k", X, field_linear)
        + 0.5 * jnp.einsum("...d,...e,...dek->...k", X, X, field_quadratic)
    )
    xphi = xphi0 + jnp.einsum("...ks,s->...k", dxphi_ds, surface)
    xtheta = xtheta0 + jnp.einsum("...ks,s->...k", dxtheta_ds, surface)
    residual = decision[nsurface + 1] * B - jnp.sum(B * B, axis=-1)[..., None] * (
        xphi + iota * xtheta
    ) if optimize_G else G * B - jnp.sum(B * B, axis=-1)[..., None] * (
        xphi + iota * xtheta
    )
    if weight_inv_modB:
        residual = residual / jnp.sqrt(jnp.sum(B * B, axis=-1))[..., None]
    return 0.5 * jnp.sum(residual * residual) / residual.size


@pytest.mark.parametrize("weight_inv_modB", [False, True])
@pytest.mark.parametrize("optimize_G", [False, True])
def test_true_hessian_matches_independent_polynomial_field_ad(
    weight_inv_modB,
    optimize_G,
):
    fixture = _polynomial_fixture()
    inputs = fixture["inputs"]
    decision = fixture["decision"] if optimize_G else fixture["decision"][:-1]
    objective = lambda q: _polynomial_objective(
        q,
        fixture["constants"],
        weight_inv_modB=weight_inv_modB,
        optimize_G=optimize_G,
    )
    expected_value, expected_gradient = jax.value_and_grad(objective)(decision)
    expected_hessian = jax.hessian(objective)(decision)
    value, gradient, hessian = boozer_residual_analytic_value_grad_hessian(
        **inputs,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        quadrature_tile_size=4,
    )

    np.testing.assert_allclose(value, expected_value, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=8e-12, atol=8e-12)


@pytest.mark.parametrize("weight_inv_modB", [False, True])
def test_value_gradient_preserves_near_root_residual_scale(weight_inv_modB):
    rng = np.random.default_rng(9)
    nphi, ntheta, nsurface = 2, 2, 2
    B = rng.normal(size=(nphi, ntheta, 3)) + np.asarray([0.0, 0.0, 2.0])
    G = 1.4
    tang = G * B / np.sum(B * B, axis=-1)[..., None]
    tang = tang + 1e-12 * rng.normal(size=tang.shape)
    zeros_surface = np.zeros((nphi, ntheta, 3, nsurface))
    inputs = {
        "G": G,
        "iota": 0.0,
        "B": B,
        "dB_dX": np.zeros((nphi, ntheta, 3, 3)),
        "xphi": tang,
        "xtheta": rng.normal(size=tang.shape),
        "dx_ds": zeros_surface,
        "dxphi_ds": rng.normal(scale=0.2, size=zeros_surface.shape),
        "dxtheta_ds": np.zeros_like(zeros_surface),
    }
    value, gradient = boozer_residual_analytic_value_grad(
        **inputs,
        optimize_G=True,
        weight_inv_modB=weight_inv_modB,
    )

    surface0 = jnp.zeros((nsurface,))
    decision = jnp.concatenate((surface0, jnp.asarray([0.0, G])))
    B_jax = jnp.asarray(B)
    xphi_jax = jnp.asarray(tang)
    xtheta_jax = jnp.asarray(inputs["xtheta"])
    dxphi_jax = jnp.asarray(inputs["dxphi_ds"])

    def objective(q):
        xphi = xphi_jax + jnp.einsum("...ks,s->...k", dxphi_jax, q[:nsurface])
        residual = q[-1] * B_jax - jnp.sum(B_jax * B_jax, axis=-1)[..., None] * (
            xphi + q[-2] * xtheta_jax
        )
        if weight_inv_modB:
            residual = residual / jnp.sqrt(jnp.sum(B_jax * B_jax, axis=-1))[..., None]
        return 0.5 * jnp.sum(residual * residual) / residual.size

    expected_value, expected_gradient = jax.value_and_grad(objective)(decision)
    assert float(expected_value) > 0.0
    np.testing.assert_allclose(value, expected_value, rtol=3e-5, atol=2e-27)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=4e-5, atol=8e-16)
