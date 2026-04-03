"""Shared helpers for the real-fixture single-stage adjoint probes."""

from __future__ import annotations

import numpy as np

from simsopt.objectives.utilities import forward_backward_jax


_STREAMING_GROUP_VJP_REQUIRED = (
    "Grouped adjoint probes require res['vjp_groups']; "
    "legacy full-pytree adjoint fallback is not allowed."
)


def iter_grouped_adjoint_cotangents(jr_jax, adj: np.ndarray):
    """Yield grouped adjoint cotangents one coil block at a time."""
    booz_jax = jr_jax.boozer_surface
    iota = booz_jax.res["iota"]
    G = booz_jax.res["G"]
    vjp_groups_fn = booz_jax.res.get("vjp_groups")
    if vjp_groups_fn is None:
        raise RuntimeError(_STREAMING_GROUP_VJP_REQUIRED)
    yield from vjp_groups_fn(adj, booz_jax, iota, G)


def compute_adjoint_state(jr_jax) -> tuple[np.ndarray, float]:
    """Return the objective-consistent adjoint vector and its residual."""
    booz_jax = jr_jax.boozer_surface
    p_mat, l_mat, u_mat = booz_jax.res["PLU"]
    coil_dofs = jr_jax.biotsavart.x.copy()
    coil_set_spec = jr_jax.biotsavart.coil_set_spec_from_dofs(coil_dofs)
    dJ_ds = jr_jax._compute_dJ_ds(
        coil_set_spec,
        booz_jax.res["iota"],
        booz_jax.res["G"],
        booz_jax.res.get("weight_inv_modB", True),
    )
    adj = forward_backward_jax(p_mat, l_mat, u_mat, dJ_ds, iterative_refinement=True)
    hessian = p_mat @ l_mat @ u_mat
    residual = hessian.T @ adj - dJ_ds
    rel = float(np.linalg.norm(residual) / (np.linalg.norm(dJ_ds) + 1e-30))
    return adj, rel


def accumulate_grouped_adjoint_derivative(
    bs_jax,
    grouped_adj_cotangents,
    *,
    on_stage=None,
):
    """Project grouped adjoint cotangents incrementally to a coil ``Derivative``."""
    from simsopt._core.derivative import Derivative

    def emit(label: str, *, group_count: int) -> None:
        if on_stage is not None:
            on_stage(label, group_count=group_count)

    total_derivative = Derivative({})
    grouped_iter = iter(grouped_adj_cotangents)
    emit("before_grouped_adjoint_vjp", group_count=0)
    try:
        current_entry = next(grouped_iter)
    except StopIteration:
        emit("after_grouped_adjoint_vjp_end", group_count=0)
        emit("after_derivative_projection", group_count=0)
        return total_derivative

    group_count = 0
    emit("after_grouped_adjoint_vjp_first_group", group_count=1)
    while True:
        try:
            next_entry = next(grouped_iter)
        except StopIteration:
            emit("after_grouped_adjoint_vjp_end", group_count=group_count + 1)
            next_entry = None

        d_coil_array, coil_group_indices = current_entry
        total_derivative += bs_jax.coil_cotangents_to_derivative(
            [d_coil_array],
            [coil_group_indices],
        )
        group_count += 1
        if next_entry is None:
            emit("after_derivative_projection", group_count=group_count)
            return total_derivative
        current_entry = next_entry


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
