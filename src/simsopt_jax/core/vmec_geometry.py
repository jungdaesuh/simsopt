"""Pure JAX VMEC geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.core.vmec_frozen import VmecFrozenSplineState, vmec_spline_eval

__all__ = [
    "VMEC_GEOMETRY_RESULT_FIELD_NAMES",
    "VmecGeometryResultsJAX",
    "vmec_compute_geometry_jax",
]


@dataclass(frozen=True)
class VmecGeometryResultsJAX:
    """JAX pytree counterpart of ``VmecGeometryResults``."""

    ns: int
    ntheta: int
    nphi: int
    s: jax.Array
    iota: jax.Array
    d_iota_d_s: jax.Array
    d_pressure_d_s: jax.Array
    shat: jax.Array
    phi: jax.Array
    theta_vmec: jax.Array
    theta_pest: jax.Array
    d_lambda_d_s: jax.Array
    d_lambda_d_theta_vmec: jax.Array
    d_lambda_d_phi: jax.Array
    sqrt_g_vmec: jax.Array
    sqrt_g_vmec_alt: jax.Array
    modB: jax.Array
    d_B_d_s: jax.Array
    d_B_d_theta_vmec: jax.Array
    d_B_d_phi: jax.Array
    B_sup_theta_vmec: jax.Array
    B_sup_theta_pest: jax.Array
    B_sup_phi: jax.Array
    B_sub_s: jax.Array
    B_sub_theta_vmec: jax.Array
    B_sub_phi: jax.Array
    edge_toroidal_flux_over_2pi: jax.Array
    sinphi: jax.Array
    cosphi: jax.Array
    d2_R_d_phi2: jax.Array
    d2_R_d_theta_vmec2: jax.Array
    d2_R_d_theta_vmec_d_phi: jax.Array
    d2_R_d_s_d_theta_vmec: jax.Array
    d2_R_d_s_d_phi: jax.Array
    d2_Z_d_theta_vmec2: jax.Array
    d2_Z_d_phi2: jax.Array
    d2_Z_d_theta_vmec_d_phi: jax.Array
    d2_Z_d_s_d_theta_vmec: jax.Array
    d2_Z_d_s_d_phi: jax.Array
    d_B_sup_phi_d_theta_vmec: jax.Array
    d_B_sup_phi_d_phi: jax.Array
    d_B_sup_theta_vmec_d_theta_vmec: jax.Array
    d_B_sup_theta_vmec_d_phi: jax.Array
    d_B_sup_theta_vmec_d_s: jax.Array
    d_B_sup_phi_d_s: jax.Array
    R: jax.Array
    d_R_d_s: jax.Array
    d_R_d_theta_vmec: jax.Array
    d_R_d_phi: jax.Array
    X: jax.Array
    Y: jax.Array
    Z: jax.Array
    d_Z_d_s: jax.Array
    d_Z_d_theta_vmec: jax.Array
    d_Z_d_phi: jax.Array
    d_X_d_theta_vmec: jax.Array
    d_X_d_phi: jax.Array
    d_X_d_s: jax.Array
    d_Y_d_theta_vmec: jax.Array
    d_Y_d_phi: jax.Array
    d_Y_d_s: jax.Array
    grad_s_X: jax.Array
    grad_s_Y: jax.Array
    grad_s_Z: jax.Array
    grad_theta_vmec_X: jax.Array
    grad_theta_vmec_Y: jax.Array
    grad_theta_vmec_Z: jax.Array
    grad_phi_X: jax.Array
    grad_phi_Y: jax.Array
    grad_phi_Z: jax.Array
    grad_psi_X: jax.Array
    grad_psi_Y: jax.Array
    grad_psi_Z: jax.Array
    grad_alpha_X: jax.Array
    grad_alpha_Y: jax.Array
    grad_alpha_Z: jax.Array
    grad_B_X: jax.Array
    grad_B_Y: jax.Array
    grad_B_Z: jax.Array
    B_X: jax.Array
    B_Y: jax.Array
    B_Z: jax.Array
    grad_s_dot_grad_s: jax.Array
    B_cross_grad_s_dot_grad_alpha: jax.Array
    B_cross_grad_s_dot_grad_alpha_alternate: jax.Array
    B_cross_grad_B_dot_grad_alpha: jax.Array
    B_cross_grad_B_dot_grad_alpha_alternate: jax.Array
    B_cross_grad_B_dot_grad_psi: jax.Array
    B_cross_kappa_dot_grad_psi: jax.Array
    B_cross_kappa_dot_grad_alpha: jax.Array
    grad_alpha_dot_grad_alpha: jax.Array
    grad_alpha_dot_grad_psi: jax.Array
    grad_psi_dot_grad_psi: jax.Array
    L_reference: jax.Array
    B_reference: jax.Array
    toroidal_flux_sign: jax.Array
    bmag: jax.Array
    gradpar_theta_pest: jax.Array
    gradpar_phi: jax.Array
    gds2: jax.Array
    gds21: jax.Array
    gds22: jax.Array
    gbdrift: jax.Array
    gbdrift0: jax.Array
    cvdrift: jax.Array
    cvdrift0: jax.Array
    grad_B__XX: jax.Array
    grad_B__XY: jax.Array
    grad_B__XZ: jax.Array
    grad_B__YX: jax.Array
    grad_B__YY: jax.Array
    grad_B__YZ: jax.Array
    grad_B__ZX: jax.Array
    grad_B__ZY: jax.Array
    grad_B__ZZ: jax.Array
    grad_B_double_dot_grad_B: jax.Array
    norm_grad_B: jax.Array
    L_grad_B: jax.Array
    nalpha: int | None = None
    nl: int | None = None
    alpha: jax.Array | None = None
    theta1d: jax.Array | None = None
    phi1d: jax.Array | None = None


VMEC_GEOMETRY_RESULT_META_FIELDS = ("ns", "ntheta", "nphi", "nalpha", "nl")

VMEC_GEOMETRY_RESULT_DATA_FIELDS = (
    "s",
    "iota",
    "d_iota_d_s",
    "d_pressure_d_s",
    "shat",
    "phi",
    "theta_vmec",
    "theta_pest",
    "d_lambda_d_s",
    "d_lambda_d_theta_vmec",
    "d_lambda_d_phi",
    "sqrt_g_vmec",
    "sqrt_g_vmec_alt",
    "modB",
    "d_B_d_s",
    "d_B_d_theta_vmec",
    "d_B_d_phi",
    "B_sup_theta_vmec",
    "B_sup_theta_pest",
    "B_sup_phi",
    "B_sub_s",
    "B_sub_theta_vmec",
    "B_sub_phi",
    "edge_toroidal_flux_over_2pi",
    "sinphi",
    "cosphi",
    "d2_R_d_phi2",
    "d2_R_d_theta_vmec2",
    "d2_R_d_theta_vmec_d_phi",
    "d2_R_d_s_d_theta_vmec",
    "d2_R_d_s_d_phi",
    "d2_Z_d_theta_vmec2",
    "d2_Z_d_phi2",
    "d2_Z_d_theta_vmec_d_phi",
    "d2_Z_d_s_d_theta_vmec",
    "d2_Z_d_s_d_phi",
    "d_B_sup_phi_d_theta_vmec",
    "d_B_sup_phi_d_phi",
    "d_B_sup_theta_vmec_d_theta_vmec",
    "d_B_sup_theta_vmec_d_phi",
    "d_B_sup_theta_vmec_d_s",
    "d_B_sup_phi_d_s",
    "R",
    "d_R_d_s",
    "d_R_d_theta_vmec",
    "d_R_d_phi",
    "X",
    "Y",
    "Z",
    "d_Z_d_s",
    "d_Z_d_theta_vmec",
    "d_Z_d_phi",
    "d_X_d_theta_vmec",
    "d_X_d_phi",
    "d_X_d_s",
    "d_Y_d_theta_vmec",
    "d_Y_d_phi",
    "d_Y_d_s",
    "grad_s_X",
    "grad_s_Y",
    "grad_s_Z",
    "grad_theta_vmec_X",
    "grad_theta_vmec_Y",
    "grad_theta_vmec_Z",
    "grad_phi_X",
    "grad_phi_Y",
    "grad_phi_Z",
    "grad_psi_X",
    "grad_psi_Y",
    "grad_psi_Z",
    "grad_alpha_X",
    "grad_alpha_Y",
    "grad_alpha_Z",
    "grad_B_X",
    "grad_B_Y",
    "grad_B_Z",
    "B_X",
    "B_Y",
    "B_Z",
    "grad_s_dot_grad_s",
    "B_cross_grad_s_dot_grad_alpha",
    "B_cross_grad_s_dot_grad_alpha_alternate",
    "B_cross_grad_B_dot_grad_alpha",
    "B_cross_grad_B_dot_grad_alpha_alternate",
    "B_cross_grad_B_dot_grad_psi",
    "B_cross_kappa_dot_grad_psi",
    "B_cross_kappa_dot_grad_alpha",
    "grad_alpha_dot_grad_alpha",
    "grad_alpha_dot_grad_psi",
    "grad_psi_dot_grad_psi",
    "L_reference",
    "B_reference",
    "toroidal_flux_sign",
    "bmag",
    "gradpar_theta_pest",
    "gradpar_phi",
    "gds2",
    "gds21",
    "gds22",
    "gbdrift",
    "gbdrift0",
    "cvdrift",
    "cvdrift0",
    "grad_B__XX",
    "grad_B__XY",
    "grad_B__XZ",
    "grad_B__YX",
    "grad_B__YY",
    "grad_B__YZ",
    "grad_B__ZX",
    "grad_B__ZY",
    "grad_B__ZZ",
    "grad_B_double_dot_grad_B",
    "norm_grad_B",
    "L_grad_B",
    "alpha",
    "theta1d",
    "phi1d",
)

VMEC_GEOMETRY_RESULT_FIELD_NAMES = (
    "ns",
    "ntheta",
    "nphi",
    *VMEC_GEOMETRY_RESULT_DATA_FIELDS[:-3],
    "nalpha",
    "nl",
    *VMEC_GEOMETRY_RESULT_DATA_FIELDS[-3:],
)

jax.tree_util.register_dataclass(
    VmecGeometryResultsJAX,
    data_fields=VMEC_GEOMETRY_RESULT_DATA_FIELDS,
    meta_fields=VMEC_GEOMETRY_RESULT_META_FIELDS,
)


def _as_rank1(values) -> jax.Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    if array.ndim == 0:
        return array[None]
    return array


def _theta_phi_tensors(
    s: jax.Array, theta_vmec, phi
) -> tuple[jax.Array, jax.Array, int, int, int]:
    theta_vmec_array = jnp.asarray(theta_vmec, dtype=s.dtype)
    phi_array = jnp.asarray(phi, dtype=s.dtype)
    if theta_vmec_array.ndim == 0:
        theta_vmec_array = theta_vmec_array[None]
    if phi_array.ndim == 0:
        phi_array = phi_array[None]
    if theta_vmec_array.ndim not in (1, 3):
        raise ValueError("theta_vmec must be a float, 1d array, or 3d array.")
    if phi_array.ndim not in (1, 3):
        raise ValueError("phi must be a float, 1d array, or 3d array.")

    ns = s.shape[0]
    ntheta = theta_vmec_array.shape[0]
    if theta_vmec_array.ndim == 3:
        ntheta = theta_vmec_array.shape[1]
    nphi = phi_array.shape[0]
    if phi_array.ndim == 3:
        nphi = phi_array.shape[2]

    if theta_vmec_array.ndim == 1:
        theta_vmec_array = jnp.broadcast_to(
            theta_vmec_array.reshape(1, ntheta, 1), (ns, ntheta, nphi)
        )
    if phi_array.ndim == 1:
        phi_array = jnp.broadcast_to(phi_array.reshape(1, 1, nphi), (ns, ntheta, nphi))
    return theta_vmec_array, phi_array, int(ns), int(ntheta), int(nphi)


def _mode_sum(coeff_cos: jax.Array, coeff_sin: jax.Array, cosangle, sinangle):
    return jnp.einsum("ij,jikl->ikl", coeff_cos, cosangle) + jnp.einsum(
        "ij,jikl->ikl", coeff_sin, sinangle
    )


def vmec_compute_geometry_jax(
    frozen_state: VmecFrozenSplineState,
    s,
    theta_vmec,
    phi,
    phi_center: float = 0.0,
) -> VmecGeometryResultsJAX:
    """Compute VMEC geometry diagnostics from frozen spline data.

    The JAX kernel preserves the CPU diagnostic formulas. AD consumers should
    stay away from the magnetic-axis ``s = 0`` drift-normalization singularity
    and points where ``grad_B_double_dot_grad_B`` vanishes unless a caller adds a
    model-specific limiting treatment.
    """
    s_array = _as_rank1(s)
    theta_vmec_array, phi_array, ns, ntheta, nphi = _theta_phi_tensors(
        s_array, theta_vmec, phi
    )

    d_pressure_d_s = vmec_spline_eval(frozen_state.d_pressure_d_s, s_array)
    iota = vmec_spline_eval(frozen_state.iota, s_array)
    d_iota_d_s = vmec_spline_eval(frozen_state.d_iota_d_s, s_array)
    shat = (-2.0 * s_array / iota) * d_iota_d_s

    rmnc = vmec_spline_eval(frozen_state.rmnc, s_array)
    zmns = vmec_spline_eval(frozen_state.zmns, s_array)
    lmns = vmec_spline_eval(frozen_state.lmns, s_array)
    d_rmnc_d_s = vmec_spline_eval(frozen_state.d_rmnc_d_s, s_array)
    d_zmns_d_s = vmec_spline_eval(frozen_state.d_zmns_d_s, s_array)
    d_lmns_d_s = vmec_spline_eval(frozen_state.d_lmns_d_s, s_array)
    rmns = vmec_spline_eval(frozen_state.rmns, s_array)
    zmnc = vmec_spline_eval(frozen_state.zmnc, s_array)
    lmnc = vmec_spline_eval(frozen_state.lmnc, s_array)
    d_rmns_d_s = vmec_spline_eval(frozen_state.d_rmns_d_s, s_array)
    d_zmnc_d_s = vmec_spline_eval(frozen_state.d_zmnc_d_s, s_array)
    d_lmnc_d_s = vmec_spline_eval(frozen_state.d_lmnc_d_s, s_array)

    gmnc = vmec_spline_eval(frozen_state.gmnc, s_array)
    bmnc = vmec_spline_eval(frozen_state.bmnc, s_array)
    d_bmnc_d_s = vmec_spline_eval(frozen_state.d_bmnc_d_s, s_array)
    bsupumnc = vmec_spline_eval(frozen_state.bsupumnc, s_array)
    bsupvmnc = vmec_spline_eval(frozen_state.bsupvmnc, s_array)
    bsubsmns = vmec_spline_eval(frozen_state.bsubsmns, s_array)
    bsubumnc = vmec_spline_eval(frozen_state.bsubumnc, s_array)
    bsubvmnc = vmec_spline_eval(frozen_state.bsubvmnc, s_array)
    d_bsupumnc_d_s = vmec_spline_eval(frozen_state.d_bsupumnc_d_s, s_array)
    d_bsupvmnc_d_s = vmec_spline_eval(frozen_state.d_bsupvmnc_d_s, s_array)
    gmns = vmec_spline_eval(frozen_state.gmns, s_array)
    bmns = vmec_spline_eval(frozen_state.bmns, s_array)
    d_bmns_d_s = vmec_spline_eval(frozen_state.d_bmns_d_s, s_array)
    bsupumns = vmec_spline_eval(frozen_state.bsupumns, s_array)
    bsupvmns = vmec_spline_eval(frozen_state.bsupvmns, s_array)
    bsubsmnc = vmec_spline_eval(frozen_state.bsubsmnc, s_array)
    bsubumns = vmec_spline_eval(frozen_state.bsubumns, s_array)
    bsubvmns = vmec_spline_eval(frozen_state.bsubvmns, s_array)
    d_bsupumns_d_s = vmec_spline_eval(frozen_state.d_bsupumns_d_s, s_array)
    d_bsupvmns_d_s = vmec_spline_eval(frozen_state.d_bsupvmns_d_s, s_array)

    angle = (
        frozen_state.xm[:, None, None, None] * theta_vmec_array[None, :, :, :]
        - frozen_state.xn[:, None, None, None] * phi_array[None, :, :, :]
    )
    cosangle = jnp.cos(angle)
    sinangle = jnp.sin(angle)
    xm = frozen_state.xm[None, :]
    xn = frozen_state.xn[None, :]
    xm2 = xm * xm
    xn2 = xn * xn
    xmxn = xm * xn

    R = _mode_sum(rmnc, rmns, cosangle, sinangle)
    d_R_d_s = _mode_sum(d_rmnc_d_s, d_rmns_d_s, cosangle, sinangle)
    d_R_d_theta_vmec = _mode_sum(-rmnc * xm, rmns * xm, sinangle, cosangle)
    d_R_d_phi = _mode_sum(rmnc * xn, -rmns * xn, sinangle, cosangle)
    d2_R_d_phi2 = _mode_sum(-rmnc * xn2, -rmns * xn2, cosangle, sinangle)
    d2_R_d_theta_vmec2 = _mode_sum(-rmnc * xm2, -rmns * xm2, cosangle, sinangle)
    d2_R_d_theta_vmec_d_phi = _mode_sum(
        rmnc * xmxn, rmns * xmxn, cosangle, sinangle
    )
    d2_R_d_s_d_theta_vmec = _mode_sum(
        -d_rmnc_d_s * xm, d_rmns_d_s * xm, sinangle, cosangle
    )
    d2_R_d_s_d_phi = _mode_sum(
        d_rmnc_d_s * xn, -d_rmns_d_s * xn, sinangle, cosangle
    )

    Z = _mode_sum(zmns, zmnc, sinangle, cosangle)
    d_Z_d_s = _mode_sum(d_zmns_d_s, d_zmnc_d_s, sinangle, cosangle)
    d_Z_d_theta_vmec = _mode_sum(zmns * xm, -zmnc * xm, cosangle, sinangle)
    d_Z_d_phi = _mode_sum(-zmns * xn, zmnc * xn, cosangle, sinangle)
    d2_Z_d_phi2 = _mode_sum(-zmns * xn2, -zmnc * xn2, sinangle, cosangle)
    d2_Z_d_theta_vmec2 = _mode_sum(-zmns * xm2, -zmnc * xm2, sinangle, cosangle)
    d2_Z_d_theta_vmec_d_phi = _mode_sum(
        zmns * xmxn, zmnc * xmxn, sinangle, cosangle
    )
    d2_Z_d_s_d_theta_vmec = _mode_sum(
        d_zmns_d_s * xm, -d_zmnc_d_s * xm, cosangle, sinangle
    )
    d2_Z_d_s_d_phi = _mode_sum(
        -d_zmns_d_s * xn, d_zmnc_d_s * xn, cosangle, sinangle
    )

    lambd = _mode_sum(lmns, lmnc, sinangle, cosangle)
    d_lambda_d_s = _mode_sum(d_lmns_d_s, d_lmnc_d_s, sinangle, cosangle)
    d_lambda_d_theta_vmec = _mode_sum(lmns * xm, -lmnc * xm, cosangle, sinangle)
    d_lambda_d_phi = _mode_sum(-lmns * xn, lmnc * xn, cosangle, sinangle)
    theta_pest = theta_vmec_array + lambd

    angle_nyq = (
        frozen_state.xm_nyq[:, None, None, None] * theta_vmec_array[None, :, :, :]
        - frozen_state.xn_nyq[:, None, None, None] * phi_array[None, :, :, :]
    )
    cosangle_nyq = jnp.cos(angle_nyq)
    sinangle_nyq = jnp.sin(angle_nyq)
    xm_nyq = frozen_state.xm_nyq[None, :]
    xn_nyq = frozen_state.xn_nyq[None, :]

    sqrt_g_vmec = _mode_sum(gmnc, gmns, cosangle_nyq, sinangle_nyq)
    modB = _mode_sum(bmnc, bmns, cosangle_nyq, sinangle_nyq)
    d_B_d_s = _mode_sum(d_bmnc_d_s, d_bmns_d_s, cosangle_nyq, sinangle_nyq)
    d_B_d_theta_vmec = _mode_sum(
        -bmnc * xm_nyq, bmns * xm_nyq, sinangle_nyq, cosangle_nyq
    )
    d_B_d_phi = _mode_sum(
        bmnc * xn_nyq, -bmns * xn_nyq, sinangle_nyq, cosangle_nyq
    )
    B_sup_theta_vmec = _mode_sum(bsupumnc, bsupumns, cosangle_nyq, sinangle_nyq)
    B_sup_phi = _mode_sum(bsupvmnc, bsupvmns, cosangle_nyq, sinangle_nyq)
    B_sub_s = _mode_sum(bsubsmns, bsubsmnc, sinangle_nyq, cosangle_nyq)
    B_sub_theta_vmec = _mode_sum(bsubumnc, bsubumns, cosangle_nyq, sinangle_nyq)
    B_sub_phi = _mode_sum(bsubvmnc, bsubvmns, cosangle_nyq, sinangle_nyq)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi
    d_B_sup_phi_d_theta_vmec = _mode_sum(
        -bsupvmnc * xm_nyq, bsupvmns * xm_nyq, sinangle_nyq, cosangle_nyq
    )
    d_B_sup_phi_d_phi = _mode_sum(
        bsupvmnc * xn_nyq, -bsupvmns * xn_nyq, sinangle_nyq, cosangle_nyq
    )
    d_B_sup_theta_vmec_d_theta_vmec = _mode_sum(
        -bsupumnc * xm_nyq, bsupumns * xm_nyq, sinangle_nyq, cosangle_nyq
    )
    d_B_sup_theta_vmec_d_phi = _mode_sum(
        bsupumnc * xn_nyq, -bsupumns * xn_nyq, sinangle_nyq, cosangle_nyq
    )
    d_B_sup_theta_vmec_d_s = _mode_sum(
        d_bsupumnc_d_s, d_bsupumns_d_s, cosangle_nyq, sinangle_nyq
    )
    d_B_sup_phi_d_s = _mode_sum(
        d_bsupvmnc_d_s, d_bsupvmns_d_s, cosangle_nyq, sinangle_nyq
    )

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)
    edge_toroidal_flux_over_2pi = -frozen_state.phiedge / (2.0 * jnp.pi)

    sinphi = jnp.sin(phi_array)
    cosphi = jnp.cos(phi_array)
    X = R * cosphi
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    Y = R * sinphi
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    grad_s_X = (
        d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi
    ) / sqrt_g_vmec
    grad_s_Y = (
        d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi
    ) / sqrt_g_vmec
    grad_s_Z = (
        d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi
    ) / sqrt_g_vmec
    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec
    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi
    phi_shifted = phi_array - phi_center
    alpha_s_term = d_lambda_d_s - phi_shifted * d_iota_d_s[:, None, None]
    alpha_phi_term = -iota[:, None, None] + d_lambda_d_phi
    grad_alpha_X = alpha_s_term * grad_s_X
    grad_alpha_Y = alpha_s_term * grad_s_Y
    grad_alpha_Z = alpha_s_term * grad_s_Z
    grad_alpha_X += (1.0 + d_lambda_d_theta_vmec) * grad_theta_vmec_X
    grad_alpha_X += alpha_phi_term * grad_phi_X
    grad_alpha_Y += (1.0 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y
    grad_alpha_Y += alpha_phi_term * grad_phi_Y
    grad_alpha_Z += (1.0 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z
    grad_alpha_Z += alpha_phi_term * grad_phi_Z

    grad_B_X = (
        d_B_d_s * grad_s_X
        + d_B_d_theta_vmec * grad_theta_vmec_X
        + d_B_d_phi * grad_phi_X
    )
    grad_B_Y = (
        d_B_d_s * grad_s_Y
        + d_B_d_theta_vmec * grad_theta_vmec_Y
        + d_B_d_phi * grad_phi_Y
    )
    grad_B_Z = (
        d_B_d_s * grad_s_Z
        + d_B_d_theta_vmec * grad_theta_vmec_Z
        + d_B_d_phi * grad_phi_Z
    )

    B_X = (
        edge_toroidal_flux_over_2pi
        * (
            (1.0 + d_lambda_d_theta_vmec) * d_X_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Y = (
        edge_toroidal_flux_over_2pi
        * (
            (1.0 + d_lambda_d_theta_vmec) * d_Y_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec
        )
        / sqrt_g_vmec
    )
    B_Z = (
        edge_toroidal_flux_over_2pi
        * (
            (1.0 + d_lambda_d_theta_vmec) * d_Z_d_phi
            + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec
        )
        / sqrt_g_vmec
    )

    B_cross_grad_s_dot_grad_alpha = (
        B_sub_phi * (1.0 + d_lambda_d_theta_vmec)
        - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
    ) / sqrt_g_vmec
    B_cross_grad_s_dot_grad_alpha_alternate = (
        B_X * grad_s_Y * grad_alpha_Z
        + B_Y * grad_s_Z * grad_alpha_X
        + B_Z * grad_s_X * grad_alpha_Y
        - B_Z * grad_s_Y * grad_alpha_X
        - B_X * grad_s_Z * grad_alpha_Y
        - B_Y * grad_s_X * grad_alpha_Z
    )
    B_cross_grad_B_dot_grad_alpha = (
        B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None])
        + B_sub_theta_vmec * d_B_d_phi * alpha_s_term
        + B_sub_phi * d_B_d_s * (1.0 + d_lambda_d_theta_vmec)
        - B_sub_phi * d_B_d_theta_vmec * alpha_s_term
        - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None])
        - B_sub_s * d_B_d_phi * (1.0 + d_lambda_d_theta_vmec)
    ) / sqrt_g_vmec
    B_cross_grad_B_dot_grad_alpha_alternate = (
        B_X * grad_B_Y * grad_alpha_Z
        + B_Y * grad_B_Z * grad_alpha_X
        + B_Z * grad_B_X * grad_alpha_Y
        - B_Z * grad_B_Y * grad_alpha_X
        - B_X * grad_B_Z * grad_alpha_Y
        - B_Y * grad_B_X * grad_alpha_Z
    )

    grad_alpha_dot_grad_alpha = (
        grad_alpha_X * grad_alpha_X
        + grad_alpha_Y * grad_alpha_Y
        + grad_alpha_Z * grad_alpha_Z
    )
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )
    grad_psi_dot_grad_psi = (
        grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z
    )
    grad_s_dot_grad_s = grad_s_X * grad_s_X + grad_s_Y * grad_s_Y + grad_s_Z * grad_s_Z
    B_cross_grad_B_dot_grad_psi = (
        (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec)
        / sqrt_g_vmec
        * edge_toroidal_flux_over_2pi
    )
    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB
    mu_0 = 4.0 * jnp.pi * 1.0e-7
    B_cross_kappa_dot_grad_alpha = (
        B_cross_grad_B_dot_grad_alpha / modB
        + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi
    )

    L_reference = frozen_state.Aminor_p
    B_reference = (
        2.0 * jnp.abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    )
    toroidal_flux_sign = jnp.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = jnp.sqrt(s_array)
    bmag = modB / B_reference
    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB
    gradpar_phi = L_reference * B_sup_phi / modB
    gds2 = (
        grad_alpha_dot_grad_alpha * L_reference * L_reference * s_array[:, None, None]
    )
    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference
    gds22 = (
        grad_psi_dot_grad_psi
        * shat[:, None, None]
        * shat[:, None, None]
        / (
            L_reference
            * L_reference
            * B_reference
            * B_reference
            * s_array[:, None, None]
        )
    )
    gbdrift = (
        -2.0
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * B_cross_grad_B_dot_grad_alpha
        / (modB * modB * modB)
        * toroidal_flux_sign
    )
    gbdrift0 = (
        B_cross_grad_B_dot_grad_psi
        * 2.0
        * shat[:, None, None]
        / (modB * modB * modB * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift = gbdrift - (
        2.0
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * mu_0
        * d_pressure_d_s[:, None, None]
        * toroidal_flux_sign
        / (edge_toroidal_flux_over_2pi * modB * modB)
    )
    cvdrift0 = gbdrift0

    d_B_X_d_s = (
        d_B_sup_theta_vmec_d_s * d_R_d_theta_vmec * cosphi
        + B_sup_theta_vmec * d2_R_d_s_d_theta_vmec * cosphi
        + d_B_sup_phi_d_s * d_R_d_phi * cosphi
        + B_sup_phi * d2_R_d_s_d_phi * cosphi
        - d_B_sup_phi_d_s * R * sinphi
        - B_sup_phi * d_R_d_s * sinphi
    )
    d_B_X_d_theta = (
        d_B_sup_theta_vmec_d_theta_vmec * d_R_d_theta_vmec * cosphi
        + B_sup_theta_vmec * d2_R_d_theta_vmec2 * cosphi
        + d_B_sup_phi_d_theta_vmec * d_R_d_phi * cosphi
        + B_sup_phi * d2_R_d_theta_vmec_d_phi * cosphi
        - d_B_sup_phi_d_theta_vmec * R * sinphi
        - B_sup_phi * d_R_d_theta_vmec * sinphi
    )
    d_B_X_d_phi = (
        d_B_sup_theta_vmec_d_phi * d_R_d_theta_vmec * cosphi
        + B_sup_theta_vmec * d2_R_d_theta_vmec_d_phi * cosphi
        - B_sup_theta_vmec * d_R_d_theta_vmec * sinphi
        + d_B_sup_phi_d_phi * d_R_d_phi * cosphi
        + B_sup_phi * d2_R_d_phi2 * cosphi
        - B_sup_phi * d_R_d_phi * sinphi
        - d_B_sup_phi_d_phi * R * sinphi
        - B_sup_phi * d_R_d_phi * sinphi
        - B_sup_phi * R * cosphi
    )
    d_B_Y_d_s = (
        d_B_sup_theta_vmec_d_s * d_R_d_theta_vmec * sinphi
        + B_sup_theta_vmec * d2_R_d_s_d_theta_vmec * sinphi
        + d_B_sup_phi_d_s * d_R_d_phi * sinphi
        + B_sup_phi * d2_R_d_s_d_phi * sinphi
        + d_B_sup_phi_d_s * R * cosphi
        + B_sup_phi * d_R_d_s * cosphi
    )
    d_B_Y_d_theta = (
        d_B_sup_theta_vmec_d_theta_vmec * d_R_d_theta_vmec * sinphi
        + B_sup_theta_vmec * d2_R_d_theta_vmec2 * sinphi
        + d_B_sup_phi_d_theta_vmec * d_R_d_phi * sinphi
        + B_sup_phi * d2_R_d_theta_vmec_d_phi * sinphi
        + d_B_sup_phi_d_theta_vmec * R * cosphi
        + B_sup_phi * d_R_d_theta_vmec * cosphi
    )
    d_B_Y_d_phi = (
        d_B_sup_theta_vmec_d_phi * d_R_d_theta_vmec * sinphi
        + B_sup_theta_vmec * d2_R_d_theta_vmec_d_phi * sinphi
        + B_sup_theta_vmec * d_R_d_theta_vmec * cosphi
        + d_B_sup_phi_d_phi * d_R_d_phi * sinphi
        + B_sup_phi * d2_R_d_phi2 * sinphi
        + B_sup_phi * d_R_d_phi * cosphi
        + d_B_sup_phi_d_phi * R * cosphi
        + B_sup_phi * d_R_d_phi * cosphi
        - B_sup_phi * R * sinphi
    )
    d_B_Z_d_s = (
        d_B_sup_theta_vmec_d_s * d_Z_d_theta_vmec
        + B_sup_theta_vmec * d2_Z_d_s_d_theta_vmec
        + d_B_sup_phi_d_s * d_Z_d_phi
        + B_sup_phi * d2_Z_d_s_d_phi
    )
    d_B_Z_d_theta = (
        d_B_sup_theta_vmec_d_theta_vmec * d_Z_d_theta_vmec
        + B_sup_theta_vmec * d2_Z_d_theta_vmec2
        + d_B_sup_phi_d_theta_vmec * d_Z_d_phi
        + B_sup_phi * d2_Z_d_theta_vmec_d_phi
    )
    d_B_Z_d_phi = (
        d_B_sup_theta_vmec_d_phi * d_Z_d_theta_vmec
        + B_sup_theta_vmec * d2_Z_d_theta_vmec_d_phi
        + d_B_sup_phi_d_phi * d_Z_d_phi
        + B_sup_phi * d2_Z_d_phi2
    )

    grad_B__XX = (
        d_B_X_d_s * grad_s_X
        + d_B_X_d_theta * grad_theta_vmec_X
        + d_B_X_d_phi * grad_phi_X
    )
    grad_B__XY = (
        d_B_X_d_s * grad_s_Y
        + d_B_X_d_theta * grad_theta_vmec_Y
        + d_B_X_d_phi * grad_phi_Y
    )
    grad_B__XZ = (
        d_B_X_d_s * grad_s_Z
        + d_B_X_d_theta * grad_theta_vmec_Z
        + d_B_X_d_phi * grad_phi_Z
    )
    grad_B__YX = (
        d_B_Y_d_s * grad_s_X
        + d_B_Y_d_theta * grad_theta_vmec_X
        + d_B_Y_d_phi * grad_phi_X
    )
    grad_B__YY = (
        d_B_Y_d_s * grad_s_Y
        + d_B_Y_d_theta * grad_theta_vmec_Y
        + d_B_Y_d_phi * grad_phi_Y
    )
    grad_B__YZ = (
        d_B_Y_d_s * grad_s_Z
        + d_B_Y_d_theta * grad_theta_vmec_Z
        + d_B_Y_d_phi * grad_phi_Z
    )
    grad_B__ZX = (
        d_B_Z_d_s * grad_s_X
        + d_B_Z_d_theta * grad_theta_vmec_X
        + d_B_Z_d_phi * grad_phi_X
    )
    grad_B__ZY = (
        d_B_Z_d_s * grad_s_Y
        + d_B_Z_d_theta * grad_theta_vmec_Y
        + d_B_Z_d_phi * grad_phi_Y
    )
    grad_B__ZZ = (
        d_B_Z_d_s * grad_s_Z
        + d_B_Z_d_theta * grad_theta_vmec_Z
        + d_B_Z_d_phi * grad_phi_Z
    )
    grad_B_double_dot_grad_B = (
        grad_B__XX * grad_B__XX
        + grad_B__XY * grad_B__XY
        + grad_B__XZ * grad_B__XZ
        + grad_B__YX * grad_B__YX
        + grad_B__YY * grad_B__YY
        + grad_B__YZ * grad_B__YZ
        + grad_B__ZX * grad_B__ZX
        + grad_B__ZY * grad_B__ZY
        + grad_B__ZZ * grad_B__ZZ
    )
    norm_grad_B = jnp.sqrt(grad_B_double_dot_grad_B)
    L_grad_B = modB * jnp.sqrt(2.0 / grad_B_double_dot_grad_B)

    return VmecGeometryResultsJAX(
        ns=ns,
        ntheta=ntheta,
        nphi=nphi,
        s=s_array,
        iota=iota,
        d_iota_d_s=d_iota_d_s,
        d_pressure_d_s=d_pressure_d_s,
        shat=shat,
        theta_vmec=theta_vmec_array,
        phi=phi_array,
        theta_pest=theta_pest,
        d_lambda_d_s=d_lambda_d_s,
        d_lambda_d_theta_vmec=d_lambda_d_theta_vmec,
        d_lambda_d_phi=d_lambda_d_phi,
        sqrt_g_vmec=sqrt_g_vmec,
        sqrt_g_vmec_alt=sqrt_g_vmec_alt,
        modB=modB,
        d_B_d_s=d_B_d_s,
        d_B_d_theta_vmec=d_B_d_theta_vmec,
        d_B_d_phi=d_B_d_phi,
        B_sup_theta_vmec=B_sup_theta_vmec,
        B_sup_theta_pest=B_sup_theta_pest,
        B_sup_phi=B_sup_phi,
        B_sub_s=B_sub_s,
        B_sub_theta_vmec=B_sub_theta_vmec,
        B_sub_phi=B_sub_phi,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
        sinphi=sinphi,
        cosphi=cosphi,
        d2_R_d_phi2=d2_R_d_phi2,
        d2_R_d_theta_vmec2=d2_R_d_theta_vmec2,
        d2_R_d_theta_vmec_d_phi=d2_R_d_theta_vmec_d_phi,
        d2_R_d_s_d_theta_vmec=d2_R_d_s_d_theta_vmec,
        d2_R_d_s_d_phi=d2_R_d_s_d_phi,
        d2_Z_d_theta_vmec2=d2_Z_d_theta_vmec2,
        d2_Z_d_phi2=d2_Z_d_phi2,
        d2_Z_d_theta_vmec_d_phi=d2_Z_d_theta_vmec_d_phi,
        d2_Z_d_s_d_theta_vmec=d2_Z_d_s_d_theta_vmec,
        d2_Z_d_s_d_phi=d2_Z_d_s_d_phi,
        d_B_sup_phi_d_theta_vmec=d_B_sup_phi_d_theta_vmec,
        d_B_sup_phi_d_phi=d_B_sup_phi_d_phi,
        d_B_sup_theta_vmec_d_theta_vmec=d_B_sup_theta_vmec_d_theta_vmec,
        d_B_sup_theta_vmec_d_phi=d_B_sup_theta_vmec_d_phi,
        d_B_sup_theta_vmec_d_s=d_B_sup_theta_vmec_d_s,
        d_B_sup_phi_d_s=d_B_sup_phi_d_s,
        R=R,
        d_R_d_s=d_R_d_s,
        d_R_d_theta_vmec=d_R_d_theta_vmec,
        d_R_d_phi=d_R_d_phi,
        X=X,
        Y=Y,
        Z=Z,
        d_Z_d_s=d_Z_d_s,
        d_Z_d_theta_vmec=d_Z_d_theta_vmec,
        d_Z_d_phi=d_Z_d_phi,
        d_X_d_theta_vmec=d_X_d_theta_vmec,
        d_X_d_phi=d_X_d_phi,
        d_X_d_s=d_X_d_s,
        d_Y_d_theta_vmec=d_Y_d_theta_vmec,
        d_Y_d_phi=d_Y_d_phi,
        d_Y_d_s=d_Y_d_s,
        grad_s_X=grad_s_X,
        grad_s_Y=grad_s_Y,
        grad_s_Z=grad_s_Z,
        grad_theta_vmec_X=grad_theta_vmec_X,
        grad_theta_vmec_Y=grad_theta_vmec_Y,
        grad_theta_vmec_Z=grad_theta_vmec_Z,
        grad_phi_X=grad_phi_X,
        grad_phi_Y=grad_phi_Y,
        grad_phi_Z=grad_phi_Z,
        grad_psi_X=grad_psi_X,
        grad_psi_Y=grad_psi_Y,
        grad_psi_Z=grad_psi_Z,
        grad_alpha_X=grad_alpha_X,
        grad_alpha_Y=grad_alpha_Y,
        grad_alpha_Z=grad_alpha_Z,
        grad_B_X=grad_B_X,
        grad_B_Y=grad_B_Y,
        grad_B_Z=grad_B_Z,
        B_X=B_X,
        B_Y=B_Y,
        B_Z=B_Z,
        grad_s_dot_grad_s=grad_s_dot_grad_s,
        B_cross_grad_s_dot_grad_alpha=B_cross_grad_s_dot_grad_alpha,
        B_cross_grad_s_dot_grad_alpha_alternate=(
            B_cross_grad_s_dot_grad_alpha_alternate
        ),
        B_cross_grad_B_dot_grad_alpha=B_cross_grad_B_dot_grad_alpha,
        B_cross_grad_B_dot_grad_alpha_alternate=(
            B_cross_grad_B_dot_grad_alpha_alternate
        ),
        B_cross_grad_B_dot_grad_psi=B_cross_grad_B_dot_grad_psi,
        B_cross_kappa_dot_grad_psi=B_cross_kappa_dot_grad_psi,
        B_cross_kappa_dot_grad_alpha=B_cross_kappa_dot_grad_alpha,
        grad_alpha_dot_grad_alpha=grad_alpha_dot_grad_alpha,
        grad_alpha_dot_grad_psi=grad_alpha_dot_grad_psi,
        grad_psi_dot_grad_psi=grad_psi_dot_grad_psi,
        L_reference=L_reference,
        B_reference=B_reference,
        toroidal_flux_sign=toroidal_flux_sign,
        bmag=bmag,
        gradpar_theta_pest=gradpar_theta_pest,
        gradpar_phi=gradpar_phi,
        gds2=gds2,
        gds21=gds21,
        gds22=gds22,
        gbdrift=gbdrift,
        gbdrift0=gbdrift0,
        cvdrift=cvdrift,
        cvdrift0=cvdrift0,
        grad_B__XX=grad_B__XX,
        grad_B__XY=grad_B__XY,
        grad_B__XZ=grad_B__XZ,
        grad_B__YX=grad_B__YX,
        grad_B__YY=grad_B__YY,
        grad_B__YZ=grad_B__YZ,
        grad_B__ZX=grad_B__ZX,
        grad_B__ZY=grad_B__ZY,
        grad_B__ZZ=grad_B__ZZ,
        grad_B_double_dot_grad_B=grad_B_double_dot_grad_B,
        norm_grad_B=norm_grad_B,
        L_grad_B=L_grad_B,
    )
