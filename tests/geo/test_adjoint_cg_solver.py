"""Tests for the opt-in adjoint linear solvers in ``optimizer.py``.

Three opt-in adjoint paths are covered here:

1. Matrix-free CG (``SIMSOPT_ADJOINT_LINEAR_SOLVER="cg"``): the inner-Boozer
   Gauss-Newton adjoint operator ``J^T J + stab I`` is symmetric
   positive-(semi)definite, so ``lineax`` Conjugate Gradients is valid.  These
   tests verify the matrix-free CG path agrees with a direct solve and with the
   established dense ``lstsq`` path, and that
   ``_solve_hessian_least_squares_system_with_status`` dispatches to CG when the
   module-level solver selector is ``"cg"``.

2. Experimental LSMR-on-J (``SIMSOPT_ADJOINT_LINEAR_SOLVER="lsmr_j"``): solve
   a positively regularized normal system through the residual Jacobian
   ``[J; sqrt(stab) I]``. This is a comparator for the future unsquared LS
   adjoint path, not the default production route.

3. Dense-LU exact-adjoint (``SIMSOPT_EXACT_ADJOINT_DENSE_LU=1``): direct LU
   factorization of the un-squared, well-conditioned ``J^T`` for the
   exact-Jacobian transpose solve (the GMRES baseline stagnates there).  These
   tests verify the LU+IR solver against an independent ``np.linalg.solve``
   oracle, the 1-D/2-D batched RHS contract, the dispatch gate
   (flag-and-transpose only), and the numerical-singularity fail-closed guard.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
lineax = pytest.importorskip("lineax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simsopt_jax.geo.optimizers import optimizer as _optimizer

_LINEAX_LSMR_AVAILABLE = hasattr(lineax, "LSMR")
_LINEAX_LSMR_SKIP_REASON = "lineax>=0.1.1 is required for LSMR comparator tests"


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
    assert _optimizer._ADJOINT_LINEAR_SOLVER != "lsmr_j"


# --- Experimental LSMR-on-J comparator for regularized normal systems ---


def _rectangular_j_problem(m=18, n=9, seed=0, stab=0.25):
    """Full-rank rectangular J, a decision-space RHS, and direct oracle."""
    rng = np.random.default_rng(seed)
    jacobian_np = rng.standard_normal((m, n))
    jacobian_np += np.linspace(0.5, 1.5, m)[:, None] * np.eye(m, n)
    rhs_np = rng.standard_normal(n)
    jacobian = jnp.asarray(jacobian_np)
    rhs = jnp.asarray(rhs_np)

    def residual_fn(x):
        return jacobian @ x

    def objective_fn(x):
        residual = residual_fn(x)
        return 0.5 * jnp.vdot(residual, residual).real

    expected = np.linalg.solve(
        jacobian_np.T @ jacobian_np + stab * np.eye(n),
        rhs_np,
    )
    return jacobian, residual_fn, objective_fn, rhs, expected


@pytest.mark.skipif(not _LINEAX_LSMR_AVAILABLE, reason=_LINEAX_LSMR_SKIP_REASON)
def test_lsmr_j_regularized_normal_system_matches_dense_oracle():
    """LSMR-on-J solves the same regularized normal system as a dense oracle."""
    _, residual_fn, _, rhs, expected = _rectangular_j_problem(seed=10)
    operator = _optimizer._jacobian_linear_operator(residual_fn, jnp.zeros(rhs.shape))
    solution, status = _optimizer._solve_regularized_normal_system_lsmr_j_with_status(
        operator,
        rhs,
        stab=0.25,
        tol=1e-11,
    )
    assert bool(status.success)
    assert int(np.asarray(status.iterations)) > 0
    np.testing.assert_allclose(
        np.asarray(solution),
        expected,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.skipif(not _LINEAX_LSMR_AVAILABLE, reason=_LINEAX_LSMR_SKIP_REASON)
def test_lsmr_j_regularized_normal_system_handles_column_batched_rhs():
    """The LSMR-on-J comparator preserves the existing column-batched RHS contract."""
    jacobian, residual_fn, _, rhs, _ = _rectangular_j_problem(seed=11)
    rng = np.random.default_rng(12)
    rhs_batched_np = np.column_stack(
        [np.asarray(rhs), rng.standard_normal(rhs.shape[0])]
    )
    rhs_batched = jnp.asarray(rhs_batched_np)
    operator = _optimizer._jacobian_linear_operator(residual_fn, jnp.zeros(rhs.shape))
    solution, status = _optimizer._solve_regularized_normal_system_lsmr_j_with_status(
        operator,
        rhs_batched,
        stab=0.25,
        tol=1e-11,
    )
    expected = np.linalg.solve(
        np.asarray(jacobian).T @ np.asarray(jacobian) + 0.25 * np.eye(rhs.shape[0]),
        rhs_batched_np,
    )
    assert bool(status.success)
    np.testing.assert_allclose(
        np.asarray(solution),
        expected,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.skipif(not _LINEAX_LSMR_AVAILABLE, reason=_LINEAX_LSMR_SKIP_REASON)
def test_hessian_least_squares_dispatches_to_lsmr_j(monkeypatch):
    """The explicit selector routes through residual-J LSMR, not the Hessian helper."""
    _, residual_fn, objective_fn, rhs, expected = _rectangular_j_problem(seed=13)
    x = jnp.zeros(rhs.shape)
    monkeypatch.setattr(_optimizer, "_ADJOINT_LINEAR_SOLVER", "lsmr_j")
    solution, status = _optimizer._solve_hessian_least_squares_system_with_status(
        objective_fn,
        x,
        rhs,
        stab=0.25,
        tol=1e-11,
        residual_fn=residual_fn,
    )
    assert bool(status.success)
    np.testing.assert_allclose(
        np.asarray(solution),
        expected,
        rtol=1e-8,
        atol=1e-10,
    )


def test_lsmr_j_dispatch_requires_residual_fn_and_positive_stab(monkeypatch):
    """The selector fails closed instead of disguising a Hessian solve as LSMR-on-J."""
    _, residual_fn, objective_fn, rhs, _ = _rectangular_j_problem(seed=14)
    x = jnp.zeros(rhs.shape)
    monkeypatch.setattr(_optimizer, "_ADJOINT_LINEAR_SOLVER", "lsmr_j")

    with pytest.raises(ValueError, match="requires a residual_fn"):
        _optimizer._solve_hessian_least_squares_system_with_status(
            objective_fn,
            x,
            rhs,
            stab=0.25,
            tol=1e-11,
        )

    with pytest.raises(ValueError, match="requires positive newton_stab"):
        _optimizer._solve_hessian_least_squares_system_with_status(
            objective_fn,
            x,
            rhs,
            stab=0.0,
            tol=1e-11,
            residual_fn=residual_fn,
        )


def test_lsmr_j_reports_lineax_dependency_contract(monkeypatch):
    """A stale Lineax install fails closed with the repo dependency contract."""
    monkeypatch.delattr(_optimizer.lineax, "LSMR", raising=False)
    with pytest.raises(RuntimeError, match="lineax>=0.1.1"):
        _optimizer._lineax_lsmr_solver(rtol=1e-11, atol=1e-11)


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


def _float32_forward_error_problem():
    """Consistent nonnormal system in the fp32 old/new condition-threshold band."""
    n = 16
    rng = np.random.default_rng(15)
    left, _ = np.linalg.qr(rng.standard_normal((n, n)))
    right, _ = np.linalg.qr(rng.standard_normal((n, n)))
    singular_values = np.geomspace(1.0, 1.0e-5, n)
    matrix_np = (left @ np.diag(singular_values) @ right.T).astype(np.float32)
    true_solution_np = rng.standard_normal(n).astype(np.float32)
    rhs_np = (matrix_np @ true_solution_np).astype(np.float32)
    matrix = jnp.asarray(matrix_np, dtype=jnp.float32)
    rhs = jnp.asarray(rhs_np, dtype=jnp.float32)

    def matvec(v):
        return matrix @ v

    return matrix, matvec, rhs, true_solution_np


def test_dense_lu_solver_matches_numpy_solve_nonsymmetric():
    """The committed LU+IR solver matches an independent dense oracle on a
    non-symmetric operator (the regime where the GMRES baseline stagnates)."""
    matrix, matvec, rhs = _nonsymmetric_problem(seed=0)
    solution, status = (
        _optimizer._solve_dense_square_operator_lu_system_with_status(
            matvec, rhs, tol=1e-12
        )
    )
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
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
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs_batched))
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
    expected = np.linalg.solve(np.asarray(matrix).T, np.asarray(rhs))
    np.testing.assert_allclose(
        np.asarray(solution), np.asarray(expected), rtol=1e-10, atol=1e-12
    )

    # The no-status adapter path is what AdjointSolveState.solve_transpose calls;
    # it must share the same dispatch gate as the status-returning helper.
    direct_solution = _optimizer._solve_jacobian_operator(
        operator, rhs, transpose=True, tol=1e-12
    )
    assert calls["n"] == 2
    np.testing.assert_allclose(
        np.asarray(direct_solution), np.asarray(expected), rtol=1e-10, atol=1e-12
    )

    # flag ON + transpose=False -> transpose-only gate must not take the LU branch.
    calls["n"] = 0
    _optimizer._solve_jacobian_operator_with_status(
        operator, rhs, transpose=False, tol=1e-12
    )
    assert calls["n"] == 0
    _optimizer._solve_jacobian_operator(operator, rhs, transpose=False, tol=1e-12)
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
    _optimizer._solve_jacobian_operator(operator, rhs, transpose=True, tol=1e-12)
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


def test_dense_condition_estimate_preserves_float32_under_transfer_guard():
    """Float32 condition estimates stay dtype-stable and strict-transfer clean."""
    matrix = jnp.diag(
        jnp.asarray(np.geomspace(1.0, 1.0e-5, 32), dtype=jnp.float32)
    )

    with jax.transfer_guard("disallow"):
        estimate = _optimizer._dense_matrix_condition_estimate(matrix)

    assert estimate.dtype == jnp.float32
    assert float(np.asarray(estimate)) == pytest.approx(1.0e5, rel=1.0e-5)


def test_float32_dense_lu_status_accepts_smoke_tolerance_operator():
    """A moderately conditioned float32 solve must not fail solely because
    the fp64 rank-tolerance rule was applied with fp32 epsilon."""
    n = 128
    matrix = jnp.diag(
        jnp.asarray(np.geomspace(1.0, 1.0e-5, n), dtype=jnp.float32)
    )
    legacy_rank_threshold = 1.0 / (n * np.finfo(np.float32).eps)
    rhs = jnp.asarray(np.random.default_rng(21).standard_normal(n), dtype=jnp.float32)

    def matvec(v):
        return matrix @ v

    with jax.transfer_guard("disallow"):
        estimate = _optimizer._dense_matrix_condition_estimate(matrix)
        solution, status = _optimizer._solve_dense_square_operator_lu_system_with_status(
            matvec,
            rhs,
            tol=1.0e-4,
        )

    assert float(np.asarray(estimate)) > legacy_rank_threshold
    assert bool(status.success)
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    solution_relative_error = np.linalg.norm(
        np.asarray(solution, dtype=np.float32) - expected
    ) / np.linalg.norm(expected)
    assert solution_relative_error < 1.0e-4


def test_float32_dense_lu_rejects_nonnormal_forward_error_through_dispatch(
    monkeypatch,
):
    """The exact-adjoint dispatch must reject fp32 solves with wrong forward error."""
    matrix, matvec, rhs, true_solution = _float32_forward_error_problem()
    operator = {"matvec": matvec, "transpose_matvec": matvec}
    monkeypatch.setattr(_optimizer, "_EXACT_ADJOINT_DENSE_LU", True)

    solution, status = _optimizer._solve_jacobian_operator_with_status(
        operator,
        rhs,
        transpose=True,
        tol=1.0e-4,
    )

    relative_error = np.linalg.norm(
        np.asarray(solution, dtype=np.float32) - true_solution
    ) / np.linalg.norm(true_solution)
    assert relative_error > 1.0e-4
    assert not bool(status.success)


def test_solve_jacobian_operator_returns_nan_when_dense_lu_status_fails(
    monkeypatch,
):
    """The solution-only exact-adjoint helper must not leak a failed LU vector."""
    matrix, matvec, rhs, true_solution = _float32_forward_error_problem()
    operator = {"matvec": matvec, "transpose_matvec": matvec}
    monkeypatch.setattr(_optimizer, "_EXACT_ADJOINT_DENSE_LU", True)

    status_solution, status = _optimizer._solve_jacobian_operator_with_status(
        operator,
        rhs,
        transpose=True,
        tol=1.0e-4,
    )
    direct_solution = _optimizer._solve_jacobian_operator(
        operator,
        rhs,
        transpose=True,
        tol=1.0e-4,
    )

    relative_error = np.linalg.norm(
        np.asarray(status_solution, dtype=np.float32) - true_solution
    ) / np.linalg.norm(true_solution)
    assert relative_error > 1.0e-4
    assert not bool(status.success)
    assert np.all(np.isnan(np.asarray(direct_solution)))


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
