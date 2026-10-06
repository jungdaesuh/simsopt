"""Hessian adjoint linear solves and least-squares routing.

This layer consumes generic kernels from :mod:`linear_solve`. It never
imports the optimizer facade.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.geo.optimizers.linear_solve import (
    _apply_column_batched_operator,
    _dense_square_operator_materialization_allowed,
    _hessian_vector_product_fn,
    _linear_solve_status,
    _place_like_concrete_array,
    _solve_dense_square_operator_least_squares_system_with_status,
    _solve_square_array_system_operator_only,
)


def _require_tree_first_leaf(tree, *, detail):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError(detail)
    return jnp.asarray(leaves[0])


def adjoint_hessian_stabilization(newton_stabilization: float | jax.Array) -> float:
    """Return the stabilization owned by the final/adjoint linearization: 0.0.

    Newton damping changes iteration directions, not the accepted-state
    Hessian, so ``newton_stabilization`` never reaches the adjoint operator.
    """
    del newton_stabilization
    return 0.0


_EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS = 2


def _hessian_linear_operator(objective_fn, x, *, stab=0.0):
    hvp_fn = _hessian_vector_product_fn(objective_fn)
    first_leaf = _require_tree_first_leaf(
        x,
        detail="Hessian linear operator state must contain at least one leaf.",
    )
    dtype = first_leaf.dtype
    decision_size = int(np.asarray(jnp.asarray(x).size))
    stab_value = _staged_like(x, stab, dtype=dtype)

    def matvec_column(v):
        return hvp_fn(x, v) + stab_value * v

    def matvec(v):
        return _apply_column_batched_operator(matvec_column, v)

    return {
        "kind": "hessian",
        "shape": (decision_size, decision_size),
        "dtype": dtype,
        "matvec": matvec,
        "transpose_matvec": matvec,
    }


def _solve_hessian_system(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    rhs = jnp.asarray(rhs)
    x = _place_like_concrete_array(x, rhs)
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    solution, _ = _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )
    return solution


def _solve_hessian_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    rhs = jnp.asarray(rhs)
    x = _place_like_concrete_array(x, rhs)
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    return _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )


def _solve_hessian_least_squares_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
):
    """Solve a Hessian adjoint system without forming normal equations."""
    rhs = jnp.asarray(rhs)
    x = _place_like_concrete_array(x, rhs)
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    if _dense_square_operator_materialization_allowed(rhs):
        return _solve_dense_square_operator_least_squares_system_with_status(
            operator["matvec"],
            rhs,
            tol=tol,
        )
    solution, status = _solve_square_array_system_operator_only(
        operator["matvec"],
        rhs,
        tol=tol,
    )
    primal_residual = rhs - operator["matvec"](solution)
    return solution, _linear_solve_status(
        solution,
        primal_residual,
        rhs,
        tol=tol,
        iterations=status.iterations,
    )
