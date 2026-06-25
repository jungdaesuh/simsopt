"""Tests for the opt-in adjoint linear solvers in ``optimizer.py``.

Two opt-in adjoint paths are covered here:

1. Matrix-free CG (``SIMSOPT_ADJOINT_LINEAR_SOLVER="cg"``): the inner-Boozer
   Gauss-Newton adjoint operator ``J^T J + stab I`` is symmetric
   positive-(semi)definite, so ``lineax`` Conjugate Gradients is valid.  These
   tests verify the matrix-free CG path agrees with a direct solve and with the
   established dense ``lstsq`` path, and that
   ``_solve_hessian_least_squares_system_with_status`` dispatches to CG when the
   module-level solver selector is ``"cg"``.

2. Dense-LU exact-adjoint (``SIMSOPT_EXACT_ADJOINT_DENSE_LU=1``): direct LU
   factorization of the un-squared, well-conditioned ``J^T`` for the
   exact-Jacobian transpose solve (the GMRES baseline stagnates there).  These
   tests verify the LU+IR solver against an independent ``np.linalg.solve``
   oracle, the 1-D/2-D batched RHS contract, the dispatch gate
   (flag-and-transpose only), and the numerical-singularity fail-closed guard.
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


# --- Opt-in dense-LU exact-Boozer adjoint solver (SIMSOPT_EXACT_ADJOINT_DENSE_LU) ---
# The exact-Jacobian adjoint solves the NON-normal ``J^T x = g`` system, which the
# restarted operator-GMRES baseline stagnates on; the opt-in path materializes the
# square ``J^T`` and solves it by direct LU + one iterative-refinement step.  These
# tests exercise the committed solver and dispatch gate directly (the existing
# adapter-gate tests patch this solver out, so they cannot detect a broken solver).


def _nonsymmetric_problem(n=12, seed=0):
    """A well-conditioned NON-symmetric matrix, its matvec, and a random rhs.

    Diagonally dominant ``randn + n*I`` is asymmetric (so it exercises the
    transpose path the dense-LU solver targets) yet well-conditioned, matching
    the production exact-Boozer ``J^T`` regime (cond ~ 1e3).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + n * np.eye(n)
    matrix = jnp.asarray(a)
    rhs = jnp.asarray(rng.standard_normal(n))

    def matvec(v):
        return matrix @ v

    return matrix, matvec, rhs


def test_dense_lu_solver_matches_numpy_solve_nonsymmetric():
    """The committed LU+IR solver matches an independent dense oracle on a
    non-symmetric operator (the regime where the GMRES baseline stagnates)."""
    matrix, matvec, rhs = _nonsymmetric_problem(seed=0)
    solution, status = (
        _optimizer._solve_dense_square_operator_lu_system_with_status(
            matvec, rhs, tol=1e-12
        )
    )
    expected = jnp.linalg.solve(matrix, rhs)
    assert bool(status.success)
    rel = float(
        np.linalg.norm(np.asarray(solution) - np.asarray(expected))
        / np.linalg.norm(np.asarray(expected))
    )
    assert rel < 1e-12, rel


def test_dense_lu_solver_batched_rhs_column_parity():
    """A 2-D ``(n, k)`` RHS is solved column-wise by one factorization; each
    column equals the single-vector solve of that column and a dense oracle."""
    matrix, matvec, _ = _nonsymmetric_problem(seed=1)
    n = matrix.shape[0]
    rng = np.random.default_rng(11)
    rhs_batched = jnp.asarray(rng.standard_normal((n, 3)))
    batched, status = (
        _optimizer._solve_dense_square_operator_lu_system_with_status(
            matvec, rhs_batched, tol=1e-12
        )
    )
    assert bool(status.success)
    expected = jnp.linalg.solve(matrix, rhs_batched)
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(expected), rtol=1e-10, atol=1e-12
    )
    for j in range(3):
        column, _ = (
            _optimizer._solve_dense_square_operator_lu_system_with_status(
                matvec, rhs_batched[:, j], tol=1e-12
            )
        )
        np.testing.assert_allclose(
            np.asarray(batched[:, j]), np.asarray(column), rtol=1e-10, atol=1e-12
        )


def test_dense_lu_materialization_gate_shape_and_byte_contract():
    """The LU materialization gate accepts 1-D and 2-D RHS, rejects 3-D, and
    rejects a RHS whose ``n x n`` materialization exceeds the byte cap."""
    allowed = _optimizer._dense_square_operator_lu_materialization_allowed
    assert allowed(jnp.zeros(8))
    assert allowed(jnp.zeros((8, 3)))
    assert not allowed(jnp.zeros((8, 3, 2)))
    # n = 40000 -> 40000^2 * 8 bytes = 12.8 GB, over any backend's dense cap.
    assert not allowed(jnp.zeros(40000))


def test_solve_jacobian_operator_gate_routes_lu_iff_flag_and_transpose(monkeypatch):
    """The committed dispatch gate routes to the dense-LU helper iff the opt-in
    flag is set AND transpose=True; flag-on + transpose=False must not take it."""
    matrix, _, rhs = _nonsymmetric_problem(seed=2)
    operator = {
        "matvec": lambda v: matrix @ v,
        "transpose_matvec": lambda v: matrix.T @ v,
    }
    calls = {"n": 0}
    real_lu = _optimizer._solve_dense_square_operator_lu_system_with_status

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_lu(*args, **kwargs)

    monkeypatch.setattr(
        _optimizer, "_solve_dense_square_operator_lu_system_with_status", spy
    )
    monkeypatch.setattr(_optimizer, "_EXACT_ADJOINT_DENSE_LU", True)

    # flag ON + transpose=True -> dense-LU solving the transpose operator.
    solution, status = _optimizer._solve_jacobian_operator_with_status(
        operator, rhs, transpose=True, tol=1e-12
    )
    assert calls["n"] == 1
    assert bool(status.success)
    expected = jnp.linalg.solve(np.asarray(matrix).T, np.asarray(rhs))
    np.testing.assert_allclose(
        np.asarray(solution), np.asarray(expected), rtol=1e-10, atol=1e-12
    )

    # flag ON + transpose=False -> transpose-only gate must not take the LU branch.
    calls["n"] = 0
    _optimizer._solve_jacobian_operator_with_status(
        operator, rhs, transpose=False, tol=1e-12
    )
    assert calls["n"] == 0


def test_solve_jacobian_operator_default_flag_does_not_use_dense_lu(monkeypatch):
    """The dense-LU branch is opt-in: at the default (flag off) the transpose
    solve does not route to the dense-LU helper."""
    assert _optimizer._EXACT_ADJOINT_DENSE_LU is False
    matrix, _, rhs = _nonsymmetric_problem(seed=4)
    operator = {
        "matvec": lambda v: matrix @ v,
        "transpose_matvec": lambda v: matrix.T @ v,
    }
    calls = {"n": 0}
    real_lu = _optimizer._solve_dense_square_operator_lu_system_with_status

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_lu(*args, **kwargs)

    monkeypatch.setattr(
        _optimizer, "_solve_dense_square_operator_lu_system_with_status", spy
    )
    _optimizer._solve_jacobian_operator_with_status(
        operator, rhs, transpose=True, tol=1e-12
    )
    assert calls["n"] == 0


def test_dense_lu_status_reports_machine_precision_residual():
    """On a well-conditioned operator the LU+IR status reports a
    machine-precision relative residual and success."""
    _, matvec, rhs = _nonsymmetric_problem(seed=5)
    _, status = _optimizer._solve_dense_square_operator_lu_system_with_status(
        matvec, rhs, tol=1e-12
    )
    assert bool(status.success)
    assert float(np.asarray(status.residual_relative)) < 1e-10


def test_dense_lu_status_fails_closed_on_singular_operator():
    """A singular operator yields a backward-stable but forward-garbage solution;
    the condition-estimate guard must fail it closed (success False) instead of
    letting a wrong adjoint flow into the gradient.

    The rhs is CONSISTENT (a @ x_true), so the singular solve has a near-zero
    residual and the backward-error gate alone would report success -- the
    nonsingular guard is what must reject it.  This keeps the test load-bearing
    (it fails if the guard is removed) rather than tautological.
    """
    n = 12
    rng = np.random.default_rng(9)
    a = rng.standard_normal((n, n))
    a[:, 0] = a[:, 1]  # exactly rank-deficient: column 0 == column 1
    matrix = jnp.asarray(a)
    rhs = jnp.asarray(a @ rng.standard_normal(n))  # consistent: residual ~ 0

    def matvec(v):
        return matrix @ v

    _, status = _optimizer._solve_dense_square_operator_lu_system_with_status(
        matvec, rhs, tol=1e-12
    )
    assert not bool(status.success)


def test_dense_lstsq_status_fails_closed_on_singular_operator():
    """The lstsq sibling shares the guard: a singular operator must fail closed
    there too.  CONSISTENT rhs (a @ x_true) keeps it load-bearing -- with an
    inconsistent rhs the least-squares residual is large and the backward-error
    gate already rejects it, which would make the test pass regardless of the
    guard (tautological)."""
    n = 12
    rng = np.random.default_rng(13)
    a = rng.standard_normal((n, n))
    a[:, 2] = a[:, 3]  # exactly rank-deficient
    matrix = jnp.asarray(a)
    rhs = jnp.asarray(a @ rng.standard_normal(n))  # consistent: residual ~ 0

    def matvec(v):
        return matrix @ v

    _, status = (
        _optimizer._solve_dense_square_operator_least_squares_system_with_status(
            matvec, rhs, tol=1e-12
        )
    )
    assert not bool(status.success)
