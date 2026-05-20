from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core.vmec_geometry import (
    VMEC_GEOMETRY_RESULT_FIELD_NAMES,
    VmecGeometryResultsJAX,
    vmec_compute_geometry_jax as vmec_compute_geometry_jax_kernel,
)
from simsopt.mhd._vmec_frozen import vmec_freeze_splines
from simsopt.mhd.vmec_diagnostics_jax import (
    vmec_compute_geometry_jax as public_vmec_compute_geometry_jax,
)
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import (
    VmecGeometryResults,
    vmec_compute_geometry,
    vmec_fieldlines,
    vmec_splines,
)

from . import TEST_DIR

_OPTIONAL_FIELDLINE_FIELDS = frozenset(("alpha", "theta1d", "phi1d"))
_GEOMETRY_ARRAY_FIELDS = tuple(
    field_name
    for field_name in VMEC_GEOMETRY_RESULT_FIELD_NAMES
    if field_name
    not in ("ns", "ntheta", "nphi", "nalpha", "nl", *_OPTIONAL_FIELDLINE_FIELDS)
)
_FIELDLINE_CONSUMED_FIELDS = (
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
)


def _vmec(filename: str) -> Vmec:
    return Vmec(os.path.join(TEST_DIR, filename))


def _assert_matching_metadata(cpu_results, jax_results: VmecGeometryResultsJAX) -> None:
    assert jax_results.ns == cpu_results.ns
    assert jax_results.ntheta == cpu_results.ntheta
    assert jax_results.nphi == cpu_results.nphi
    assert jax_results.nalpha is None
    assert jax_results.nl is None
    assert jax_results.alpha is None
    assert jax_results.theta1d is None
    assert jax_results.phi1d is None


def _assert_fields_close(
    cpu_results, jax_results: VmecGeometryResultsJAX, fields
) -> None:
    for field_name in fields:
        np.testing.assert_allclose(
            np.asarray(getattr(jax_results, field_name)),
            getattr(cpu_results, field_name),
            rtol=1e-10,
            atol=1e-12,
            err_msg=field_name,
        )


def _geometry_transfer_guard_token(results: VmecGeometryResultsJAX):
    token = jnp.asarray(0.0, dtype=jnp.float64)
    for field_name in (
        "modB",
        "L_grad_B",
        "grad_alpha_dot_grad_alpha",
        "B_cross_grad_B_dot_grad_alpha",
        "B_cross_kappa_dot_grad_psi",
    ):
        token = token + jnp.sum(jnp.asarray(getattr(results, field_name)))
    return token


def test_vmec_geometry_result_jax_fields_match_cpu_dataclass():
    assert tuple(
        field.name for field in dataclasses.fields(VmecGeometryResultsJAX)
    ) == tuple(field.name for field in dataclasses.fields(VmecGeometryResults))
    assert VMEC_GEOMETRY_RESULT_FIELD_NAMES == tuple(
        field.name for field in dataclasses.fields(VmecGeometryResults)
    )


def test_vmec_compute_geometry_jax_stellsym_matches_cpu():
    """Oracle: CPU ``vmec_compute_geometry`` on pinned ``wout_li383`` fixture.

    Lane: direct_kernel VMEC geometry parity, rtol=1e-10, atol=1e-12.
    """
    vmec = _vmec("wout_li383_low_res_reference.nc")
    frozen = vmec_freeze_splines(vmec)
    s = np.array([0.25, 0.75])
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, 5, endpoint=False)

    cpu_results = vmec_compute_geometry(vmec, s, theta, phi)
    jax_results = vmec_compute_geometry_jax_kernel(
        frozen, jnp.asarray(s), jnp.asarray(theta), jnp.asarray(phi)
    )

    _assert_matching_metadata(cpu_results, jax_results)
    _assert_fields_close(cpu_results, jax_results, _GEOMETRY_ARRAY_FIELDS)


def test_vmec_compute_geometry_jax_accepts_fieldline_theta():
    """Oracle: CPU ``vmec_compute_geometry`` using CPU fieldline theta grid.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel
    fieldline-consumed geometry parity, rtol=1e-10, atol=1e-12.
    """
    vmec = _vmec("wout_li383_low_res_reference.nc")
    frozen = vmec_freeze_splines(vmec)
    s = np.array([0.25, 0.75])
    alpha = np.array([0.0, np.pi / 7.0])
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    fieldline_results = vmec_fieldlines(vmec, s, alpha, theta1d=theta)

    cpu_results = vmec_compute_geometry(
        vmec, s, fieldline_results.theta_vmec, fieldline_results.phi
    )
    jax_results = vmec_compute_geometry_jax_kernel(
        frozen,
        jnp.asarray(s),
        jnp.asarray(fieldline_results.theta_vmec),
        jnp.asarray(fieldline_results.phi),
    )

    _assert_matching_metadata(cpu_results, jax_results)
    _assert_fields_close(cpu_results, jax_results, _FIELDLINE_CONSUMED_FIELDS)


def test_vmec_compute_geometry_jax_non_stellsym_lmnc_terms_match_cpu():
    """Oracle: CPU ``vmec_compute_geometry`` on pinned non-stellsym VMEC data.

    Dataset: ``wout_10x10.nc``. Lane: direct_kernel asymmetric geometry parity,
    rtol=1e-10, atol=1e-12.
    """
    vmec = _vmec("wout_10x10.nc")
    frozen = vmec_freeze_splines(vmec)
    s = np.array([0.2, 0.7])
    theta = np.linspace(-0.6 * np.pi, 0.8 * np.pi, 4)
    phi = np.linspace(0.1, 2.0 * np.pi / vmec.wout.nfp, 5)

    cpu_results = vmec_compute_geometry(vmec, s, theta, phi)
    jax_results = vmec_compute_geometry_jax_kernel(
        frozen, jnp.asarray(s), jnp.asarray(theta), jnp.asarray(phi)
    )

    _assert_matching_metadata(cpu_results, jax_results)
    _assert_fields_close(cpu_results, jax_results, _GEOMETRY_ARRAY_FIELDS)


def test_vmec_compute_geometry_jax_nonzero_phi_center_matches_cpu():
    """Oracle: CPU geometry with nonzero ``phi_center`` on pinned VMEC data.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel
    fieldline-consumed geometry parity, rtol=1e-10, atol=1e-12.
    """
    vmec = _vmec("wout_li383_low_res_reference.nc")
    frozen = vmec_freeze_splines(vmec)
    s = np.array([0.3, 0.6])
    theta = np.linspace(-0.7 * np.pi, 0.5 * np.pi, 4)
    phi = np.linspace(0.05, 2.0 * np.pi / vmec.wout.nfp, 5)
    phi_center = 0.17

    cpu_results = vmec_compute_geometry(vmec, s, theta, phi, phi_center=phi_center)
    jax_results = vmec_compute_geometry_jax_kernel(
        frozen,
        jnp.asarray(s),
        jnp.asarray(theta),
        jnp.asarray(phi),
        phi_center=phi_center,
    )

    _assert_matching_metadata(cpu_results, jax_results)
    _assert_fields_close(cpu_results, jax_results, _FIELDLINE_CONSUMED_FIELDS)


def test_vmec_compute_geometry_jax_transfer_guard_clean():
    vmec = _vmec("wout_li383_low_res_reference.nc")
    frozen = vmec_freeze_splines(vmec)
    s = jnp.asarray([0.4])
    theta = jnp.asarray(np.linspace(-np.pi, np.pi, 3, endpoint=False))
    phi = jnp.asarray(np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, 4, endpoint=False))
    compiled = jax.jit(
        lambda frozen_arg, s_arg, theta_arg, phi_arg: _geometry_transfer_guard_token(
            vmec_compute_geometry_jax_kernel(frozen_arg, s_arg, theta_arg, phi_arg)
        )
    )

    compiled(frozen, s, theta, phi).block_until_ready()
    with jax.transfer_guard("disallow"):
        compiled(frozen, s, theta, phi).block_until_ready()


def test_vmec_compute_geometry_jax_frozen_coeff_gradient_matches_finite_difference():
    vmec = _vmec("wout_li383_low_res_reference.nc")
    frozen = vmec_freeze_splines(vmec)
    s = jnp.asarray([0.37])
    theta = jnp.asarray(np.linspace(-np.pi, np.pi, 3, endpoint=False))
    phi = jnp.asarray(np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, 4, endpoint=False))
    weights = jnp.linspace(0.3, 1.4, 12, dtype=jnp.float64).reshape(1, 3, 4)

    def weighted_modB(bmnc_coeffs):
        perturbed_frozen = dataclasses.replace(
            frozen,
            bmnc=dataclasses.replace(frozen.bmnc, coeffs=bmnc_coeffs),
        )
        geometry = vmec_compute_geometry_jax_kernel(perturbed_frozen, s, theta, phi)
        return jnp.sum(weights * geometry.modB)

    coeffs = frozen.bmnc.coeffs
    gradient = jax.grad(weighted_modB)(coeffs)
    rng = np.random.default_rng(20260519)
    direction_np = rng.normal(size=np.asarray(coeffs).shape)
    direction_np = direction_np / np.linalg.norm(direction_np)
    direction = jnp.asarray(direction_np)
    step = 1e-6
    finite_difference = (
        weighted_modB(coeffs + step * direction)
        - weighted_modB(coeffs - step * direction)
    ) / (2.0 * step)
    directional_derivative = jnp.vdot(gradient, direction)

    np.testing.assert_allclose(
        np.asarray(finite_difference),
        np.asarray(directional_derivative),
        rtol=1e-8,
        atol=1e-10,
    )


def test_public_vmec_compute_geometry_jax_accepts_vmec_splines_and_frozen_state():
    """Oracle: CPU geometry across public VMEC/spline/frozen input forms.

    Dataset: ``wout_li383_low_res_reference.nc``. Lane: direct_kernel public
    wrapper parity, rtol=1e-10, atol=1e-12.
    """
    vmec = _vmec("wout_li383_low_res_reference.nc")
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)
    s = np.array([0.35, 0.8])
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, 5, endpoint=False)
    cpu_results = vmec_compute_geometry(vmec, s, theta, phi)

    for source in (vmec, splines, frozen):
        jax_results = public_vmec_compute_geometry_jax(
            source, jnp.asarray(s), jnp.asarray(theta), jnp.asarray(phi)
        )
        _assert_matching_metadata(cpu_results, jax_results)
        _assert_fields_close(
            cpu_results, jax_results, ("theta_pest", "modB", "L_grad_B")
        )


def test_public_vmec_compute_geometry_jax_lazy_mhd_export():
    import simsopt.mhd as mhd

    vmec = _vmec("wout_li383_low_res_reference.nc")
    s = np.array([0.35])
    theta = np.linspace(-np.pi, np.pi, 3, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / vmec.wout.nfp, 4, endpoint=False)
    expected = vmec_compute_geometry(vmec, s, theta, phi)
    actual = mhd.vmec_compute_geometry_jax(
        vmec,
        jnp.asarray(s),
        jnp.asarray(theta),
        jnp.asarray(phi),
    )

    _assert_fields_close(expected, actual, ("theta_pest", "modB", "L_grad_B"))
