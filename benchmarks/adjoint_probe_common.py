"""Shared helpers for the real-fixture single-stage adjoint probes."""

from __future__ import annotations

import numpy as np


_STREAMING_GROUP_VJP_REQUIRED = (
    "Grouped adjoint probes require a valid Boozer runtime adjoint state; "
    "legacy full-pytree adjoint fallback is not allowed."
)


def iter_grouped_adjoint_cotangents(jr_jax, adj: np.ndarray):
    """Yield grouped adjoint cotangents one coil block at a time."""
    booz_jax = jr_jax.boozer_surface
    adjoint_state = booz_jax.get_adjoint_runtime_state()
    yield from adjoint_state.stream_group_vjps(adj)


def compute_adjoint_state(jr_jax) -> tuple[np.ndarray, float]:
    """Return the objective-consistent adjoint vector and its residual."""
    booz_jax = jr_jax.boozer_surface
    adjoint_state = booz_jax.get_adjoint_runtime_state()
    solved_state = adjoint_state.solved_state
    coil_dofs = jr_jax.biotsavart.x.copy()
    coil_set_spec = jr_jax.biotsavart.coil_set_spec_from_dofs(coil_dofs)
    dJ_ds = jr_jax._compute_dJ_ds(
        coil_set_spec,
        solved_state.iota,
        solved_state.G,
        solved_state.weight_inv_modB,
        sdofs=solved_state.sdofs,
    )
    solve_with_status = getattr(adjoint_state, "solve_transpose_with_status", None)
    if callable(solve_with_status):
        adj, success = solve_with_status(dJ_ds)
        if not bool(np.asarray(success)):
            raise RuntimeError(
                "Grouped adjoint probe failed because the operator-backed "
                f"transpose solve did not converge ({adjoint_state.linearization_kind})."
            )
    else:
        adj = adjoint_state.solve_transpose(dJ_ds)
    residual = adjoint_state.apply_transpose(adj) - dJ_ds
    rel = float(np.linalg.norm(residual) / (np.linalg.norm(dJ_ds) + 1e-30))
    return adj, rel


def accumulate_grouped_adjoint_dofs_gradient(
    bs_jax,
    grouped_adj_cotangents,
    *,
    coil_dofs=None,
    on_stage=None,
):
    """Project grouped adjoint cotangents incrementally to flat coil DOFs."""
    import jax.numpy as jnp

    def emit(label: str, *, group_count: int) -> None:
        if on_stage is not None:
            on_stage(label, group_count=group_count)

    if coil_dofs is None:
        coil_dofs = bs_jax.x.copy()
    coil_dofs = jnp.asarray(coil_dofs)
    total_gradient = jnp.zeros_like(coil_dofs)
    grouped_iter = iter(grouped_adj_cotangents)
    emit("before_grouped_adjoint_vjp", group_count=0)
    try:
        current_entry = next(grouped_iter)
    except StopIteration:
        emit("after_grouped_adjoint_vjp_end", group_count=0)
        emit("after_dofs_gradient_projection", group_count=0)
        return total_gradient

    group_count = 0
    emit("after_grouped_adjoint_vjp_first_group", group_count=1)
    while True:
        try:
            next_entry = next(grouped_iter)
        except StopIteration:
            emit("after_grouped_adjoint_vjp_end", group_count=group_count + 1)
            next_entry = None

        d_coil_array, coil_group_indices = current_entry
        total_gradient = total_gradient + bs_jax.coil_cotangents_to_dofs_gradient(
            [d_coil_array],
            [coil_group_indices],
            coil_dofs=coil_dofs,
        )
        group_count += 1
        if next_entry is None:
            emit("after_dofs_gradient_projection", group_count=group_count)
            return total_gradient
        current_entry = next_entry


def compute_gradient_l2_metrics(gradient) -> tuple[float, bool]:
    """Compute finite-ness and L2 norm for a flat coil-DOF gradient."""
    import jax
    import jax.numpy as jnp

    gradient = jnp.asarray(gradient)
    return (
        float(jax.device_get(jnp.linalg.norm(gradient))),
        bool(jax.device_get(jnp.all(jnp.isfinite(gradient)))),
    )


def compute_derivative_l2_metrics(derivative, optim) -> tuple[float, bool]:
    """Compute finite-ness and L2 norm without materializing the full gradient.

    This mirrors the free-dof accumulation in
    :meth:`simsopt._core.derivative.Derivative.__call__` (``as_derivative=False``)
    block by block, but reads ``derivative.data`` non-mutatingly: a missing
    optimizable contributes a zero block instead of triggering the
    ``OptimizableDefaultDict`` insertion that ``self.data[opt]`` would. The
    concatenation of these per-lineage blocks is exactly ``derivative(optim)``,
    so accumulating ``dot(block, block)`` yields ``norm(full_gradient) ** 2``.
    """
    data = derivative.data
    sq_norm = 0.0
    finite = True
    for k in optim.unique_dof_lineage:
        if not np.any(k.dofs_free_status):
            continue
        local_derivs = np.zeros(k.local_dof_size)
        for opt in k.dofs.dep_opts():
            block = data.get(opt)
            if block is None:
                continue
            local_derivs += block[opt.local_dofs_free_status]
        finite = finite and bool(np.all(np.isfinite(local_derivs)))
        sq_norm += float(np.dot(local_derivs, local_derivs))
    return float(np.sqrt(sq_norm)), finite


def compute_implicit_gradient_correction(jr_jax, bs_jax, adj: np.ndarray) -> np.ndarray:
    """Project the grouped coil cotangents back to coil DOFs."""
    import jax

    adjoint_gradient = accumulate_grouped_adjoint_dofs_gradient(
        bs_jax,
        iter_grouped_adjoint_cotangents(jr_jax, adj),
    )
    return np.asarray(jax.device_get(adjoint_gradient), dtype=float)
