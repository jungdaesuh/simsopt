"""Adjoint linear-solver selection and Hessian least-squares routing.

This layer selects among dense, CG, and residual-J LSMR formulations while
consuming generic kernels from :mod:`linear_solve` and dense-IR policy from
:mod:`dense_ir`. It never imports the optimizer facade.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
import lineax
import numpy as np

from simsopt_jax.backend import get_backend_policy
from simsopt_jax.core._device_scalars import staged_like as _staged_like
from simsopt_jax.geo.optimizers.dense_ir import (
    _solve_mixed_dense_ir_operator_with_status,
    resolve_mixed_dense_ir_policy,
)
from simsopt_jax.geo.optimizers.linear_solve import (
    _LinearSolveStatus,
    _apply_column_batched_operator,
    _dense_square_operator_materialization_allowed,
    _device_int32,
    _effective_linear_solve_tolerance,
    _hessian_vector_product_fn,
    _jacobian_linear_operator,
    _linear_solve_status,
    _optimizer_scalar,
    _place_like_concrete_array,
    _solve_dense_square_operator_least_squares_system_with_status,
    _solve_square_array_system_operator_only,
)


def _require_tree_first_leaf(tree, *, detail):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError(detail)
    return jnp.asarray(leaves[0])


_AdjointHessianLinearSolver = Literal["dense", "cg", "lsmr_j"]


_ADJOINT_LINEAR_SOLVER = (
    os.environ.get("SIMSOPT_ADJOINT_LINEAR_SOLVER", "dense").strip().lower()
)


def adjoint_hessian_stabilization(
    newton_stabilization: float | jax.Array,
    *,
    solver: _AdjointHessianLinearSolver | None = None,
) -> float | jax.Array:
    """Return the stabilization owned by the final/adjoint linearization.

    Newton damping changes iteration directions, not the accepted-state
    Hessian. The residual-J LSMR formulation is the sole exception because its
    augmented operator explicitly includes the positive regularization.
    """
    selected_solver = _ADJOINT_LINEAR_SOLVER if solver is None else solver
    if selected_solver == "lsmr_j":
        return newton_stabilization
    return 0.0


_EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS = 2


def _lineax_lsmr_solver(*, rtol, atol, max_steps=None):
    """Return the Lineax LSMR solver required by residual-J comparator paths."""
    solver_type = getattr(lineax, "LSMR", None)
    if solver_type is None:
        raise RuntimeError(
            "Lineax LSMR is required for this solver path. Install the JAX "
            "extra from pyproject.toml so lineax>=0.1.1 is available."
        )
    return solver_type(rtol=rtol, atol=atol, max_steps=max_steps)


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


def _solve_regularized_normal_system_lsmr_j_with_status(
    jacobian_operator,
    rhs,
    *,
    stab,
    tol,
):
    """Solve ``(J.T @ J + stab I) x = rhs`` through augmented residuals.

    ``stab`` must be positive so the equivalent least-squares problem
    ``min_x ||[J; sqrt(stab) I] x - [0; rhs/sqrt(stab)]||`` is well-defined.
    The helper returns a decision-space vector and the established normal-system
    residual status, making it comparable to the dense and CG adjoint helpers.
    """
    rhs = jnp.asarray(rhs)
    if rhs.ndim != 1:

        def solve_column(column):
            return _solve_regularized_normal_system_lsmr_j_with_status(
                jacobian_operator,
                column,
                stab=stab,
                tol=tol,
            )

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

    stab_host = float(stab)
    if stab_host <= 0.0:
        raise ValueError(
            "SIMSOPT_ADJOINT_LINEAR_SOLVER=lsmr_j requires positive "
            "newton_stab. The unstabilized stab=0 normal system needs a "
            "separate KKT/two-solve formulation."
        )

    dtype = rhs.dtype
    residual_size, decision_size = jacobian_operator["shape"]
    sqrt_stab = jnp.sqrt(_optimizer_scalar(stab_host, dtype=dtype))
    residual_target = jnp.zeros((residual_size,), dtype=dtype)
    target = jnp.concatenate((residual_target, rhs / sqrt_stab), axis=0)

    def augmented_matvec(vector):
        residual_part = jnp.ravel(
            jnp.asarray(jacobian_operator["matvec"](vector), dtype=dtype)
        )
        return jnp.concatenate((residual_part, sqrt_stab * vector), axis=0)

    operator = lineax.FunctionLinearOperator(
        augmented_matvec,
        jax.ShapeDtypeStruct((decision_size,), dtype),
    )
    effective_tol = _effective_linear_solve_tolerance(rhs, tol)
    # LSMR stops on the augmented least-squares criterion, while callers gate the
    # induced normal-system residual below.  Ask the inner solve for a modestly
    # tighter LS tolerance so the returned solution is judged by the same status
    # contract as the dense/CG helpers.
    solver_tol = _optimizer_scalar(1.0e-4, dtype=dtype) * effective_tol
    max_steps = max(20, 10 * int(decision_size))
    solution = lineax.linear_solve(
        operator,
        target,
        solver=_lineax_lsmr_solver(
            rtol=solver_tol,
            atol=solver_tol,
            max_steps=max_steps,
        ),
        throw=False,
    )
    j_solution = jacobian_operator["matvec"](solution.value)
    normal_residual = rhs - (
        jacobian_operator["transpose_matvec"](j_solution)
        + _optimizer_scalar(stab_host, dtype=dtype) * solution.value
    )
    return solution.value, _linear_solve_status(
        solution.value,
        normal_residual,
        rhs,
        tol=tol,
        iterations=_device_int32(solution.stats["num_steps"]),
    )


def _solve_hessian_least_squares_system_with_status(
    objective_fn,
    x,
    rhs,
    *,
    stab,
    tol,
    residual_fn=None,
    proposal_objective_fn=None,
    certificate_probe_key=None,
    solver: _AdjointHessianLinearSolver | None = None,
):
    """Solve a Hessian adjoint system without forming normal equations."""
    rhs = jnp.asarray(rhs)
    x = _place_like_concrete_array(x, rhs)
    operator = _hessian_linear_operator(objective_fn, x, stab=stab)
    selected_solver = _ADJOINT_LINEAR_SOLVER if solver is None else solver
    if selected_solver == "cg":
        return _solve_symmetric_operator_cg_with_status(
            operator["matvec"],
            rhs,
            tol=tol,
        )
    if selected_solver == "lsmr_j":
        if residual_fn is None:
            raise ValueError(
                "SIMSOPT_ADJOINT_LINEAR_SOLVER=lsmr_j requires a residual_fn "
                "so it can operate on the residual Jacobian J instead of the "
                "squared Hessian operator."
            )
        return _solve_regularized_normal_system_lsmr_j_with_status(
            _jacobian_linear_operator(residual_fn, x),
            rhs,
            stab=stab,
            tol=tol,
        )
    if _dense_square_operator_materialization_allowed(rhs):
        if proposal_objective_fn is not None:
            if certificate_probe_key is None:
                raise ValueError(
                    "Mixed dense IR requires a fresh or replay-authorized "
                    "runtime certificate key."
                )
            policy = get_backend_policy()
            dense_ir_policy = resolve_mixed_dense_ir_policy()
            proposal_dtype = np.dtype(policy.compute_dtype)
            certificate_dtype = np.dtype(policy.runtime_dtype)
            if (
                proposal_dtype != np.dtype(np.float32)
                or certificate_dtype != dense_ir_policy.certificate_dtype
            ):
                raise ValueError(
                    "A proposal objective requires the FP32-factor/"
                    f"{dense_ir_policy.certificate_dtype_name}-certificate "
                    "mixed-precision policy."
                )
            proposal_operator = _hessian_linear_operator(
                proposal_objective_fn,
                jnp.asarray(x, dtype=proposal_dtype),
                stab=stab,
            )
            return _solve_mixed_dense_ir_operator_with_status(
                proposal_operator["matvec"],
                operator["matvec"],
                rhs,
                tol=tol,
                proposal_dtype=proposal_dtype,
                certificate_sweep_dtype=operator["dtype"],
                certificate_probe_key=certificate_probe_key,
            )
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
