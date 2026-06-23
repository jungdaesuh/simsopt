"""Tests for the opt-in matrix-free CG adjoint solver (SIMSOPT_ADJOINT_LINEAR_SOLVER).

The inner-Boozer Gauss-Newton adjoint operator ``J^T J + stab I`` is symmetric
positive-(semi)definite, so ``lineax`` Conjugate Gradients is a valid solver for
it.  These tests verify the matrix-free CG path agrees with a direct solve and
with the established dense ``lstsq`` path, and that
``_solve_hessian_least_squares_system_with_status`` dispatches to CG when the
module-level solver selector is ``"cg"``.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("lineax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simsopt_jax.geo.optimizers import optimizer as _optimizer


def _spd_problem(n=12, seed=0):
    """A well-conditioned SPD matrix, its matvec, and a random rhs."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    matrix = jnp.asarray(a @ a.T + n * np.eye(n))
    rhs = jnp.asarray(rng.standard_normal(n))

    def matvec(v):
        return matrix @ v

    return matrix, matvec, rhs


def test_cg_matches_direct_solve():
    matrix, matvec, rhs = _spd_problem(seed=0)
    solution, status = _optimizer._solve_symmetric_operator_cg_with_status(
        matvec, rhs, tol=1e-12
    )
    expected = jnp.linalg.solve(matrix, rhs)
    assert bool(status.success)
    assert int(np.asarray(status.iterations)) > 0
    np.testing.assert_allclose(
        np.asarray(solution), np.asarray(expected), rtol=1e-8, atol=1e-10
    )


def test_cg_matches_dense_lstsq_path():
    _, matvec, rhs = _spd_problem(seed=1)
    cg_solution, _ = _optimizer._solve_symmetric_operator_cg_with_status(
        matvec, rhs, tol=1e-12
    )
    dense_solution, _ = (
        _optimizer._solve_dense_square_operator_least_squares_system_with_status(
            matvec, rhs, tol=1e-12
        )
    )
    np.testing.assert_allclose(
        np.asarray(cg_solution), np.asarray(dense_solution), rtol=1e-8, atol=1e-10
    )


def test_cg_handles_column_batched_rhs():
    matrix, matvec, _ = _spd_problem(seed=3)
    rng = np.random.default_rng(7)
    rhs = jnp.asarray(rng.standard_normal((matrix.shape[0], 3)))
    solutions, status = _optimizer._solve_symmetric_operator_cg_with_status(
        matvec, rhs, tol=1e-12
    )
    expected = jnp.linalg.solve(matrix, rhs)
    assert bool(status.success)
    np.testing.assert_allclose(
        np.asarray(solutions), np.asarray(expected), rtol=1e-8, atol=1e-10
    )


def test_hessian_least_squares_dispatches_to_cg(monkeypatch):
    """With the selector set to 'cg', the Hessian LS solve routes through CG."""
    matrix, _, rhs = _spd_problem(seed=2)

    def objective_fn(x):
        # Hessian of 0.5 x^T A x is the SPD matrix A.
        return 0.5 * jnp.dot(x, matrix @ x)

    x = jnp.zeros(matrix.shape[0])
    expected = jnp.linalg.solve(matrix, rhs)

    monkeypatch.setattr(_optimizer, "_ADJOINT_LINEAR_SOLVER", "cg")
    solution, status = _optimizer._solve_hessian_least_squares_system_with_status(
        objective_fn, x, rhs, stab=0.0, tol=1e-12
    )
    assert bool(status.success)
    np.testing.assert_allclose(
        np.asarray(solution), np.asarray(expected), rtol=1e-8, atol=1e-10
    )


def test_default_selector_does_not_use_cg():
    """The default selector keeps the established (non-CG) path."""
    assert _optimizer._ADJOINT_LINEAR_SOLVER != "cg"
