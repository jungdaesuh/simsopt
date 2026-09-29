"""Adjoint linear-solver selection and Hessian least-squares routing.

This layer selects between the dense and CG formulations while
consuming generic kernels from :mod:`linear_solve`. It never imports the
optimizer facade.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
import lineax
import numpy as np

from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.geo.optimizers.linear_solve import (
    _LinearSolveStatus,
    _apply_column_batched_operator,
    _dense_square_operator_materialization_allowed,
    _device_int32,
    _effective_linear_solve_tolerance,
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


_AdjointHessianLinearSolver = Literal["dense", "cg"]


def _validated_adjoint_linear_solver(
    value: str, *, source: str
) -> _AdjointHessianLinearSolver:
    """Return ``value`` as a supported adjoint solver name, else raise.

    Only ``"dense"`` and ``"cg"`` exist; any other name, including the removed
    residual-Jacobian ``"lsmr_j"`` comparator, is rejected instead of silently
    running the dense solve.
    """
    if value == "dense":
        return "dense"
    if value == "cg":
        return "cg"
    raise ValueError(
        f"{source} must be 'dense' or 'cg', got {value!r}; the 'lsmr_j' "
        "residual-Jacobian adjoint solver was removed."
    )


_ADJOINT_LINEAR_SOLVER = _validated_adjoint_linear_solver(
    os.environ.get("SIMSOPT_ADJOINT_LINEAR_SOLVER", "dense").strip().lower(),
    source="SIMSOPT_ADJOINT_LINEAR_SOLVER",
)


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


def _solve_symmetric_operator_cg_with_status(matvec, rhs, *, tol):
    """Solve a symmetric PSD operator system matrix-free via ``lineax`` CG.

    For the inner-Boozer Gauss-Newton adjoint (``J^T J + stab I``), which is
    symmetric positive-(semi)definite.  Bounded memory (no dense N x N); see
    ``_ADJOINT_LINEAR_SOLVER`` for the speed/conditioning caveats.  Handles a
    1-D rhs directly and a column-batched 2-D rhs by mapping over columns,
    mirroring ``_solve_square_array_system_operator_only``.
    """
    rhs = jnp.asarray(rhs)
    if rhs.ndim != 1:

        def solve_column(column):
            return _solve_symmetric_operator_cg_with_status(matvec, column, tol=tol)

        solutions, column_statuses = jax.vmap(
            solve_column,
            in_axes=1,
            out_axes=(1, 0),
        )(rhs)
        return solutions, _LinearSolveStatus(
            success=jnp.all(column_statuses.success),
            residual=jnp.max(column_statuses.residual),
            residual_relative=jnp.max(column_statuses.residual_relative),
            iterations=jnp.max(column_statuses.iterations),
        )

    effective_tol = _effective_linear_solve_tolerance(rhs, tol)
    operator = lineax.FunctionLinearOperator(
        matvec,
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        tags=(lineax.positive_semidefinite_tag, lineax.symmetric_tag),
    )
    solution = lineax.linear_solve(
        operator,
        rhs,
        solver=lineax.CG(rtol=effective_tol, atol=effective_tol),
        throw=False,
    )
    residual = rhs - matvec(solution.value)
    iterations = _device_int32(solution.stats["num_steps"])
    return solution.value, _linear_solve_status(
        solution.value,
        residual,
        rhs,
        tol=tol,
        iterations=iterations,
    )


def _solve_hessian_least_squares_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
    solver: _AdjointHessianLinearSolver | None = None,
):
    """Solve a Hessian adjoint system without forming normal equations."""
    rhs = jnp.asarray(rhs)
    x = _place_like_concrete_array(x, rhs)
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    selected_solver = (
        _ADJOINT_LINEAR_SOLVER
        if solver is None
        else _validated_adjoint_linear_solver(solver, source="solver")
    )
    if selected_solver == "cg":
        return _solve_symmetric_operator_cg_with_status(
            operator["matvec"],
            rhs,
            tol=tol,
        )
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
