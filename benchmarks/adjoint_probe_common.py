"""Shared helpers for the real-fixture single-stage adjoint probes."""

from __future__ import annotations

import numpy as np


def compute_adjoint_state(jr_jax) -> tuple[np.ndarray, float]:
    """Return the objective-consistent adjoint vector and its residual."""
    from simsopt.objectives.utilities import forward_backward

    booz_jax = jr_jax.boozer_surface
    p_mat, l_mat, u_mat = booz_jax.res["PLU"]
    surface = jr_jax.surface
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    constraint_weight = (
        jr_jax.constraint_weight if jr_jax.constraint_weight is not None else 1.0
    )
    dJ_ds = jr_jax._compute_dJ_ds(
        booz_jax.res["iota"],
        booz_jax.res["G"],
        booz_jax.res.get("weight_inv_modB", True),
        constraint_weight,
        nphi,
        ntheta,
    )
    adj = forward_backward(p_mat, l_mat, u_mat, dJ_ds)
    hessian = p_mat @ l_mat @ u_mat
    residual = hessian.T @ adj - dJ_ds
    rel = float(np.linalg.norm(residual) / (np.linalg.norm(dJ_ds) + 1e-30))
    return adj, rel


def compute_implicit_gradient_correction(jr_jax, bs_jax, adj: np.ndarray) -> np.ndarray:
    """Project the grouped coil cotangents back to coil DOFs."""
    from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

    booz_jax = jr_jax.boozer_surface
    vjp_fn = booz_jax.res["vjp"]
    adj_cot = vjp_fn(adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"])
    adj_deriv = _coil_cotangents_to_derivative(bs_jax.coils, *adj_cot)
    return np.asarray(adj_deriv(bs_jax), dtype=float)
