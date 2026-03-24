"""Shared helpers for the real-fixture single-stage adjoint probes."""

from __future__ import annotations

import numpy as np


def iter_grouped_adjoint_cotangents(jr_jax, adj: np.ndarray):
    """Yield grouped adjoint cotangents one coil block at a time."""
    booz_jax = jr_jax.boozer_surface
    iota = booz_jax.res["iota"]
    G = booz_jax.res["G"]
    vjp_groups_fn = booz_jax.res.get("vjp_groups")
    if vjp_groups_fn is not None:
        yield from vjp_groups_fn(adj, booz_jax, iota, G)
        return

    d_coil_arrays, coil_indices = booz_jax.res["vjp"](adj, booz_jax, iota, G)
    for d_coil_array, coil_group_indices in zip(d_coil_arrays, coil_indices):
        yield d_coil_array, coil_group_indices


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


def compute_grouped_adjoint_cotangents(jr_jax, adj: np.ndarray):
    """Return grouped adjoint cotangents before projection back to coil DOFs."""
    grouped_entries = list(iter_grouped_adjoint_cotangents(jr_jax, adj))
    d_coil_arrays = [entry[0] for entry in grouped_entries]
    coil_indices = [entry[1] for entry in grouped_entries]
    return d_coil_arrays, coil_indices


def project_grouped_adjoint_derivative(bs_jax, adj_cot):
    """Project grouped adjoint cotangents to a coil ``Derivative``."""
    from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

    return _coil_cotangents_to_derivative(bs_jax.coils, *adj_cot)


def accumulate_grouped_adjoint_derivative(bs_jax, grouped_adj_cotangents):
    """Project grouped adjoint cotangents incrementally to a coil ``Derivative``."""
    from simsopt._core.derivative import Derivative
    from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

    total_derivative = Derivative({})
    for d_coil_array, coil_group_indices in grouped_adj_cotangents:
        total_derivative += _coil_cotangents_to_derivative(
            bs_jax.coils,
            [d_coil_array],
            [coil_group_indices],
        )
    return total_derivative


def compute_derivative_l2_metrics(derivative, optim) -> tuple[float, bool]:
    """Compute finite-ness and L2 norm without materializing the full gradient."""
    from simsopt._core.derivative import _iter_local_free_derivative_blocks

    sq_norm = 0.0
    finite = True
    for local_derivs in _iter_local_free_derivative_blocks(
        derivative.data,
        optim,
        populate_missing=False,
    ):
        finite = finite and bool(np.all(np.isfinite(local_derivs)))
        sq_norm += float(np.dot(local_derivs, local_derivs))
    return float(np.sqrt(sq_norm)), finite


def compute_implicit_gradient_correction(jr_jax, bs_jax, adj: np.ndarray) -> np.ndarray:
    """Project the grouped coil cotangents back to coil DOFs."""
    adj_deriv = accumulate_grouped_adjoint_derivative(
        bs_jax,
        iter_grouped_adjoint_cotangents(jr_jax, adj),
    )
    return np.asarray(adj_deriv(bs_jax), dtype=float)
