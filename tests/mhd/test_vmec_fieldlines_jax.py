import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core.vmec_fieldlines import (
    theta_vmec_from_theta_pest_implicit_jax,
    theta_vmec_from_theta_pest_scan_jax,
    theta_vmec_residual_jax,
)
from simsopt_jax.core.vmec_frozen import vmec_spline_eval
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines, vmec_splines
from simsopt_jax.mhd.vmec_diagnostics import vmec_fieldlines_jax
from simsopt_jax_adapters.mhd.vmec_diagnostics import (
    vmec_fieldlines_jax as adapter_vmec_fieldlines_jax,
    vmec_freeze_splines,
)

from . import TEST_DIR

_FIELDLINE_PUBLIC_FIELDS = (
    "theta_vmec",
    "phi",
    "theta_pest",
    "modB",
    "B_sup_theta_pest",
    "B_sup_phi",
    "B_cross_grad_B_dot_grad_alpha",
    "B_cross_grad_B_dot_grad_psi",
    "B_cross_kappa_dot_grad_alpha",
    "B_cross_kappa_dot_grad_psi",
    "grad_alpha_dot_grad_alpha",
    "grad_alpha_dot_grad_psi",
    "grad_psi_dot_grad_psi",
    "bmag",
    "gradpar_theta_pest",
    "gradpar_phi",
    "gbdrift",
    "gbdrift0",
    "cvdrift",
    "cvdrift0",
    "gds2",
    "gds21",
    "gds22",
    "L_grad_B",
)


def _frozen_fieldline_state(filename):
    vmec = Vmec(os.path.join(TEST_DIR, filename))
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)
    return splines, frozen


def _assert_public_fieldlines_match(expected, actual, fields) -> None:
    assert actual.ns == expected.ns
    assert actual.ntheta == expected.ntheta
    assert actual.nphi == expected.nphi
    assert actual.nalpha == expected.nalpha
    assert actual.nl == expected.nl
    np.testing.assert_allclose(np.asarray(actual.s), expected.s)
    np.testing.assert_allclose(np.asarray(actual.alpha), expected.alpha)
    for field_name in fields:
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field_name)),
            getattr(expected, field_name),
            rtol=1e-10,
            atol=1e-12,
            err_msg=field_name,
        )


def test_theta_vmec_from_theta_pest_scan_matches_cpu_fieldlines_theta_branch():
    """Oracle: CPU ``vmec_fieldlines`` SciPy-Newton theta branch."""
    splines, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = np.array([0.25, 0.75])
    alpha = np.array([0.0, np.pi / 3.0])
    theta1d = np.linspace(-np.pi, np.pi, 5)
    expected = vmec_fieldlines(splines, s=s, alpha=alpha, theta1d=theta1d)
    lmns = vmec_spline_eval(frozen.lmns, s)

    actual = theta_vmec_from_theta_pest_scan_jax(
        expected.theta_pest,
        expected.phi,
        frozen.xm,
        frozen.xn,
        lmns,
        max_iter=12,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.theta_vmec,
        rtol=1e-11,
        atol=1e-12,
    )


def test_theta_vmec_from_theta_pest_implicit_matches_cpu_fieldlines_phi_branch():
    """Oracle: CPU ``vmec_fieldlines`` SciPy-Newton phi branch."""
    splines, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = np.array([0.5])
    alpha = np.array([-np.pi / 2.0, np.pi / 5.0])
    phi1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, 6)
    expected = vmec_fieldlines(splines, s=s, alpha=alpha, phi1d=phi1d)
    lmns = vmec_spline_eval(frozen.lmns, s)

    actual = theta_vmec_from_theta_pest_implicit_jax(
        expected.theta_pest,
        expected.phi,
        frozen.xm,
        frozen.xn,
        lmns,
        max_iter=20,
        tol=1e-13,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.theta_vmec,
        rtol=1e-11,
        atol=1e-12,
    )


def test_theta_vmec_implicit_branch_stable_across_warm_starts():
    """Oracle: CPU fieldline theta branch for both Newton warm starts."""
    splines, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = np.array([0.25, 0.75])
    alpha = np.array([0.0, np.pi / 3.0])
    theta1d = np.linspace(-np.pi, np.pi, 5)
    expected = vmec_fieldlines(splines, s=s, alpha=alpha, theta1d=theta1d)
    lmns = vmec_spline_eval(frozen.lmns, s)

    from_theta_pest = theta_vmec_from_theta_pest_implicit_jax(
        expected.theta_pest,
        expected.phi,
        frozen.xm,
        frozen.xn,
        lmns,
        initial_theta_vmec=expected.theta_pest,
        max_iter=20,
        tol=1e-13,
    )
    from_shifted_start = theta_vmec_from_theta_pest_implicit_jax(
        expected.theta_pest,
        expected.phi,
        frozen.xm,
        frozen.xn,
        lmns,
        initial_theta_vmec=expected.theta_pest + 0.05,
        max_iter=20,
        tol=1e-13,
    )

    np.testing.assert_allclose(
        np.asarray(from_theta_pest),
        expected.theta_vmec,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(from_shifted_start),
        expected.theta_vmec,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(from_theta_pest),
        np.asarray(from_shifted_start),
        rtol=1e-11,
        atol=1e-12,
    )


def test_theta_vmec_scan_zero_iterations_returns_explicit_warm_start():
    """Oracle: fixed-scan solver uses the supplied theta_vmec initial state."""
    theta_pest = jnp.array([[[0.0, 0.3, 0.6]]], dtype=jnp.float64)
    phi = jnp.array([[[0.0, 0.2, 0.4]]], dtype=jnp.float64)
    xm = jnp.array([1.0, 2.0], dtype=jnp.float64)
    xn = jnp.array([0.0, 1.0], dtype=jnp.float64)
    lmns = jnp.array([[0.03, -0.02]], dtype=jnp.float64)
    initial = theta_pest + 0.123

    actual = theta_vmec_from_theta_pest_scan_jax(
        theta_pest,
        phi,
        xm,
        xn,
        lmns,
        initial_theta_vmec=initial,
        max_iter=0,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(initial), rtol=0, atol=0)


def test_theta_vmec_implicit_gradient_matches_finite_difference():
    """Oracle: centered FD of the implicit theta solve wrt lambda coefficients."""
    theta_pest = jnp.array([[-0.4, 0.2, 0.7]], dtype=jnp.float64).reshape(1, 1, 3)
    phi = jnp.array([[-0.2, 0.1, 0.4]], dtype=jnp.float64).reshape(1, 1, 3)
    xm = jnp.array([1.0, 2.0], dtype=jnp.float64)
    xn = jnp.array([0.0, 1.0], dtype=jnp.float64)
    lmns = jnp.array([0.04, -0.015], dtype=jnp.float64)

    def objective(coeffs):
        roots = theta_vmec_from_theta_pest_implicit_jax(
            theta_pest,
            phi,
            xm,
            xn,
            coeffs[None, :],
            max_iter=20,
            tol=1e-13,
        )
        return jnp.sum(roots)

    gradient = jax.grad(objective)(lmns)
    step = 1e-6
    basis = np.eye(lmns.size)
    finite_difference = np.array(
        [
            (
                objective(lmns + step * jnp.asarray(direction))
                - objective(lmns - step * jnp.asarray(direction))
            )
            / (2.0 * step)
            for direction in basis
        ]
    )

    np.testing.assert_allclose(
        np.asarray(gradient),
        finite_difference,
        rtol=1e-6,
        atol=1e-8,
    )


def test_theta_vmec_implicit_transfer_guard_clean():
    """Oracle: compiled fieldline theta solve runs without implicit transfers."""
    theta_pest = jnp.array([[[0.0, 0.3, 0.6]]], dtype=jnp.float64)
    phi = jnp.array([[[0.0, 0.2, 0.4]]], dtype=jnp.float64)
    xm = jnp.array([1.0, 2.0], dtype=jnp.float64)
    xn = jnp.array([0.0, 1.0], dtype=jnp.float64)
    lmns = jnp.array([[0.03, -0.02]], dtype=jnp.float64)
    compiled = jax.jit(
        lambda theta_arg, phi_arg, lmns_arg: theta_vmec_from_theta_pest_implicit_jax(
            theta_arg,
            phi_arg,
            xm,
            xn,
            lmns_arg,
            max_iter=20,
            tol=1e-13,
        )
    )
    compiled(theta_pest, phi, lmns).block_until_ready()

    with jax.transfer_guard("disallow"):
        actual = compiled(theta_pest, phi, lmns)
        actual.block_until_ready()

    residual = theta_vmec_residual_jax(
        actual[0, 0],
        theta_pest[0, 0],
        phi[0, 0],
        xm,
        xn,
        lmns[0],
    )
    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-12)


def test_public_vmec_fieldlines_jax_frozen_state_gradient_matches_finite_difference():
    """Oracle: centered FD of fieldline metric wrt frozen lambda coefficients."""
    _, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = jnp.asarray([0.37])
    alpha = jnp.asarray([0.0, jnp.pi / 7.0])
    theta1d = jnp.asarray(np.linspace(-0.75 * np.pi, 0.8 * np.pi, 4))

    def grad_psi_dot_grad_psi_norm(lmns_coeffs):
        perturbed_frozen = dataclasses.replace(
            frozen,
            lmns=dataclasses.replace(frozen.lmns, coeffs=lmns_coeffs),
        )
        results = vmec_fieldlines_jax(
            perturbed_frozen,
            s=s,
            alpha=alpha,
            theta1d=theta1d,
        )
        return jnp.linalg.norm(results.grad_psi_dot_grad_psi)

    coeffs = frozen.lmns.coeffs
    gradient = jax.grad(grad_psi_dot_grad_psi_norm)(coeffs)
    rng = np.random.default_rng(20260519)
    direction_np = rng.normal(size=np.asarray(coeffs).shape)
    direction_np = direction_np / np.linalg.norm(direction_np)
    direction = jnp.asarray(direction_np)
    step = 1e-6
    finite_difference = (
        grad_psi_dot_grad_psi_norm(coeffs + step * direction)
        - grad_psi_dot_grad_psi_norm(coeffs - step * direction)
    ) / (2.0 * step)
    directional_derivative = jnp.vdot(gradient, direction)

    np.testing.assert_allclose(
        np.asarray(finite_difference),
        np.asarray(directional_derivative),
        rtol=1e-8,
        atol=1e-10,
    )


def test_public_vmec_fieldlines_jax_theta_branch_matches_cpu():
    """Oracle: CPU ``vmec_fieldlines`` theta branch on pinned VMEC data.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel public
    fieldline parity, rtol=1e-10, atol=1e-12.
    """
    splines, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = np.array([0.25, 0.75])
    alpha = np.array([0.0, np.pi / 3.0])
    theta1d = np.linspace(-np.pi, np.pi, 5)
    phi_center = 0.17

    expected = vmec_fieldlines(
        splines,
        s=s,
        alpha=alpha,
        theta1d=theta1d,
        phi_center=phi_center,
    )
    actual = vmec_fieldlines_jax(
        frozen,
        s=s,
        alpha=alpha,
        theta1d=theta1d,
        phi_center=phi_center,
    )

    assert actual.phi1d is None
    np.testing.assert_allclose(np.asarray(actual.theta1d), theta1d)
    _assert_public_fieldlines_match(expected, actual, _FIELDLINE_PUBLIC_FIELDS)


def test_public_vmec_fieldlines_jax_phi_branch_matches_cpu():
    """Oracle: CPU ``vmec_fieldlines`` phi branch on pinned VMEC data.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel public
    fieldline parity, rtol=1e-10, atol=1e-12.
    """
    splines, frozen = _frozen_fieldline_state("wout_li383_low_res_reference.nc")
    s = np.array([0.5])
    alpha = np.array([-np.pi / 2.0, np.pi / 5.0])
    phi1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, 6)
    phi_center = -0.11

    expected = vmec_fieldlines(
        splines,
        s=s,
        alpha=alpha,
        phi1d=phi1d,
        phi_center=phi_center,
    )
    actual = vmec_fieldlines_jax(
        frozen,
        s=s,
        alpha=alpha,
        phi1d=phi1d,
        phi_center=phi_center,
    )

    assert actual.theta1d is None
    np.testing.assert_allclose(np.asarray(actual.phi1d), phi1d)
    _assert_public_fieldlines_match(expected, actual, _FIELDLINE_PUBLIC_FIELDS)


def test_adapter_vmec_fieldlines_jax_accepts_vmec_object():
    """Oracle: CPU ``vmec_fieldlines`` through adapter-owned VMEC export.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel public
    wrapper parity, rtol=1e-10, atol=1e-12.
    """
    vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
    s = 0.5
    alpha = 0.0
    theta1d = np.linspace(-np.pi, np.pi, 4)

    expected = vmec_fieldlines(vmec, s=s, alpha=alpha, theta1d=theta1d)
    actual = adapter_vmec_fieldlines_jax(vmec, s=s, alpha=alpha, theta1d=theta1d)

    np.testing.assert_allclose(np.asarray(actual.theta1d), theta1d)
    _assert_public_fieldlines_match(
        expected, actual, ("theta_vmec", "modB", "L_grad_B")
    )
