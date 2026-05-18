"""Shared pure JAX curve kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._device_scalars import float_scalar, two_pi
from ._math_utils import as_runtime_float64 as _as_runtime_float64
from .surface_rzfourier import (
    surface_rz_fourier_spec_from_dofs,
)


@jax.jit
def incremental_arclength_pure(d1gamma):
    """Return pointwise curve arclength increments."""
    return jnp.linalg.norm(d1gamma, axis=1)


@jax.jit
def kappa_pure(d1gamma, d2gamma):
    """Return pointwise curvature for first and second curve derivatives."""
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


@jax.jit
def torsion_pure(d1gamma, d2gamma, d3gamma):
    """Return pointwise torsion for first three curve derivatives."""
    cross12 = jnp.cross(d1gamma, d2gamma, axis=1)
    return jnp.sum(cross12 * d3gamma, axis=1) / jnp.sum(cross12 * cross12, axis=1)


def _selector_matrix(size, positions, *, reference):
    matrix = np.zeros((len(positions), size), dtype=np.float64)
    if positions:
        matrix[np.arange(len(positions)), positions] = 1.0
    return _as_runtime_float64(matrix, reference=reference)


def _curve_mode_selectors(order, *, reference):
    size = 4 * order + 2
    return (
        _selector_matrix(size, list(range(0, order + 1)), reference=reference),
        _selector_matrix(
            size,
            list(range(order + 1, 2 * order + 1)),
            reference=reference,
        ),
        _selector_matrix(
            size,
            list(range(2 * order + 1, 3 * order + 2)),
            reference=reference,
        ),
        _selector_matrix(
            size,
            list(range(3 * order + 2, 4 * order + 2)),
            reference=reference,
        ),
    )


def _harmonic_terms(qpts, start_mode, count, trig_fn):
    modes = _as_runtime_float64(
        np.arange(start_mode, start_mode + count, dtype=np.float64),
        reference=qpts,
    )
    angles = qpts[:, None] * two_pi(qpts) * modes[None, :]
    return trig_fn(angles)


def gamma_2d(modes, qpts, order, G: int = 0, H: int = 0):
    """Return the 2D curve-on-surface coordinates ``(phi, theta)``."""
    modes_jax = _as_runtime_float64(modes, reference=modes)
    qpts_jax = _as_runtime_float64(qpts, reference=qpts)
    phic_sel, phis_sel, thetac_sel, thetas_sel = _curve_mode_selectors(
        order,
        reference=modes_jax,
    )
    phic = phic_sel @ modes_jax
    phis = phis_sel @ modes_jax
    thetac = thetac_sel @ modes_jax
    thetas = thetas_sel @ modes_jax

    cos_terms = _harmonic_terms(qpts_jax, 0, order + 1, jnp.cos)
    sin_terms = _harmonic_terms(qpts_jax, 1, order, jnp.sin)
    theta = (
        cos_terms @ thetac
        + sin_terms @ thetas
        + jax.lax.stop_gradient(float_scalar(int(G), qpts_jax)) * qpts_jax
    )
    phi = (
        cos_terms @ phic
        + sin_terms @ phis
        + jax.lax.stop_gradient(float_scalar(int(H), qpts_jax)) * qpts_jax
    )
    return phi, theta


def _surface_rz_fourier_gamma_pointwise(surface_spec, phi_qpts, theta_qpts):
    """Evaluate an RZ Fourier surface at paired ``(phi, theta)`` samples."""
    angle_scale = two_pi(theta_qpts)
    phi = angle_scale * phi_qpts
    theta = angle_scale * theta_qpts
    m = _as_runtime_float64(
        np.arange(surface_spec.mpol + 1, dtype=np.float64),
        reference=theta_qpts,
    )
    n = _as_runtime_float64(
        np.arange(2 * surface_spec.ntor + 1, dtype=np.float64) - surface_spec.ntor,
        reference=phi_qpts,
    )
    nfp = float_scalar(surface_spec.nfp, n)
    angles = (
        m[None, :, None] * theta[:, None, None]
        - nfp * n[None, None, :] * phi[:, None, None]
    )
    cos_terms = jnp.cos(angles)
    sin_terms = jnp.sin(angles)
    radius = jnp.sum(
        surface_spec.rc[None, :, :] * cos_terms
        + surface_spec.rs[None, :, :] * sin_terms,
        axis=(1, 2),
    )
    z = jnp.sum(
        surface_spec.zc[None, :, :] * cos_terms
        + surface_spec.zs[None, :, :] * sin_terms,
        axis=(1, 2),
    )
    return jnp.stack([radius * jnp.cos(phi), radius * jnp.sin(phi), z], axis=-1)


def curve_cws_rz_gamma_from_dofs(
    curve_dofs,
    qpts,
    order,
    G,
    H,
    surf_dofs,
    mpol,
    ntor,
    nfp,
    stellsym=True,
):
    """Return 3D coordinates for a CWS Fourier curve on an RZ Fourier surface."""
    phi, theta = gamma_2d(curve_dofs, qpts, order, G, H)
    surface_spec = surface_rz_fourier_spec_from_dofs(
        _as_runtime_float64(surf_dofs, reference=curve_dofs),
        quadpoints_phi=phi,
        quadpoints_theta=theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    return _surface_rz_fourier_gamma_pointwise(surface_spec, phi, theta)
