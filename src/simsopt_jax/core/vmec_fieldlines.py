"""Pure JAX kernels for VMEC field-line coordinate inversion."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64
from ._root import _newton_scan_fixed_iters_diagonal, _newton_with_implicit_vjp_diagonal

__all__ = [
    "theta_vmec_residual_jax",
    "theta_vmec_from_theta_pest_implicit_jax",
    "theta_vmec_from_theta_pest_scan_jax",
]


def theta_vmec_residual_jax(
    theta_vmec,
    theta_pest,
    phi,
    xm,
    xn,
    lmns,
) -> jax.Array:
    """Return CPU ``vmec_fieldlines`` residual for ``theta_vmec``."""
    theta_vmec_jax = as_jax_float64(theta_vmec)
    theta_pest_jax = as_jax_float64(theta_pest)
    phi_jax = as_jax_float64(phi)
    xm_jax = as_jax_float64(xm)
    xn_jax = as_jax_float64(xn)
    lmns_jax = as_jax_float64(lmns)
    angle = (
        xm_jax[:, None] * theta_vmec_jax[None, :] - xn_jax[:, None] * phi_jax[None, :]
    )
    lambd = jnp.sum(lmns_jax[:, None] * jnp.sin(angle), axis=0)
    return theta_pest_jax - (theta_vmec_jax + lambd)


def _theta_vmec_jacobian_diagonal(theta_vmec, phi, xm, xn, lmns):
    theta_vmec_jax = as_jax_float64(theta_vmec)
    phi_jax = as_jax_float64(phi)
    xm_jax = as_jax_float64(xm)
    xn_jax = as_jax_float64(xn)
    lmns_jax = as_jax_float64(lmns)
    angle = (
        xm_jax[:, None] * theta_vmec_jax[None, :] - xn_jax[:, None] * phi_jax[None, :]
    )
    diagonal = -(
        1.0 + jnp.sum(lmns_jax[:, None] * xm_jax[:, None] * jnp.cos(angle), axis=0)
    )
    return diagonal


def _theta_vmec_line_scan(
    theta_pest, phi, xm, xn, lmns, initial_theta_vmec, max_iter: int
):
    def residual(theta_vmec):
        return theta_vmec_residual_jax(theta_vmec, theta_pest, phi, xm, xn, lmns)

    def jacobian_diagonal(theta_vmec):
        return _theta_vmec_jacobian_diagonal(theta_vmec, phi, xm, xn, lmns)

    return _newton_scan_fixed_iters_diagonal(
        residual,
        as_jax_float64(initial_theta_vmec),
        max_iter=max_iter,
        jac_diagonal=jacobian_diagonal,
    )


def _theta_vmec_line_implicit(
    theta_pest,
    phi,
    xm,
    xn,
    lmns,
    initial_theta_vmec,
    max_iter: int,
    tol: float,
):
    def residual(theta_vmec, params):
        return theta_vmec_residual_jax(
            theta_vmec,
            params["theta_pest"],
            params["phi"],
            params["xm"],
            params["xn"],
            params["lmns"],
        )

    def jacobian_diagonal(theta_vmec, params):
        return _theta_vmec_jacobian_diagonal(
            theta_vmec,
            params["phi"],
            params["xm"],
            params["xn"],
            params["lmns"],
        )

    return _newton_with_implicit_vjp_diagonal(
        residual,
        as_jax_float64(initial_theta_vmec),
        {
            "theta_pest": as_jax_float64(theta_pest),
            "phi": as_jax_float64(phi),
            "xm": as_jax_float64(xm),
            "xn": as_jax_float64(xn),
            "lmns": as_jax_float64(lmns),
        },
        max_iter=max_iter,
        tol=tol,
        jac_diagonal=jacobian_diagonal,
    )


def theta_vmec_from_theta_pest_scan_jax(
    theta_pest,
    phi,
    xm,
    xn,
    lmns,
    *,
    initial_theta_vmec=None,
    max_iter: int = 12,
) -> jax.Array:
    """Solve field-line ``theta_pest -> theta_vmec`` with fixed iterations."""
    theta_pest_jax = as_jax_float64(theta_pest)
    phi_jax = as_jax_float64(phi)
    xm_jax = as_jax_float64(xm)
    xn_jax = as_jax_float64(xn)
    lmns_jax = as_jax_float64(lmns)
    initial_theta_vmec_jax = (
        theta_pest_jax
        if initial_theta_vmec is None
        else as_jax_float64(initial_theta_vmec)
    )
    solve_alpha = jax.vmap(
        lambda theta_line, phi_line, initial_line, lmns_surface: _theta_vmec_line_scan(
            theta_line,
            phi_line,
            xm_jax,
            xn_jax,
            lmns_surface,
            initial_line,
            max_iter,
        ),
        in_axes=(0, 0, 0, None),
    )
    solve_surface = jax.vmap(
        lambda theta_surface, phi_surface, initial_surface, lmns_surface: solve_alpha(
            theta_surface,
            phi_surface,
            initial_surface,
            lmns_surface,
        ),
        in_axes=(0, 0, 0, 0),
    )
    return solve_surface(theta_pest_jax, phi_jax, initial_theta_vmec_jax, lmns_jax)


def theta_vmec_from_theta_pest_implicit_jax(
    theta_pest,
    phi,
    xm,
    xn,
    lmns,
    *,
    initial_theta_vmec=None,
    max_iter: int = 20,
    tol: float = 1e-13,
) -> jax.Array:
    """Solve field-line ``theta_pest -> theta_vmec`` with implicit-VJP roots."""
    theta_pest_jax = as_jax_float64(theta_pest)
    phi_jax = as_jax_float64(phi)
    xm_jax = as_jax_float64(xm)
    xn_jax = as_jax_float64(xn)
    lmns_jax = as_jax_float64(lmns)
    initial_theta_vmec_jax = (
        theta_pest_jax
        if initial_theta_vmec is None
        else as_jax_float64(initial_theta_vmec)
    )
    solve_alpha = jax.vmap(
        lambda theta_line, phi_line, initial_line, lmns_surface: (
            _theta_vmec_line_implicit(
                theta_line,
                phi_line,
                xm_jax,
                xn_jax,
                lmns_surface,
                initial_line,
                max_iter,
                tol,
            )
        ),
        in_axes=(0, 0, 0, None),
    )
    solve_surface = jax.vmap(
        lambda theta_surface, phi_surface, initial_surface, lmns_surface: solve_alpha(
            theta_surface,
            phi_surface,
            initial_surface,
            lmns_surface,
        ),
        in_axes=(0, 0, 0, 0),
    )
    return solve_surface(theta_pest_jax, phi_jax, initial_theta_vmec_jax, lmns_jax)
