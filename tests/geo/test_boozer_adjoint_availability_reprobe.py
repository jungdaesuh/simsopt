"""Tests for the dense-LU Boozer adjoint-availability re-probe.

When the eager least-squares-Hessian availability probe in the host-controlled
Boozer solve fails on a validly-converged surface -- the squared Hessian
(cond ``kappa(J)^2``) is too ill-conditioned to reach ``linear_solve_tol`` -- and
the dense-LU exact-adjoint opt-in is enabled, ``_reprobe_adjoint_availability_
with_dense_lu`` recovers availability via the un-squared ``J^T`` solver that
the exact-Jacobian dense-LU opt-in path uses.  It must stay inert when the flag
is off, must not mask a nonlinear-solve failure, and must skip a non-square
Jacobian.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simsopt_jax.geo.optimizers import (
    linear_solve as _linear_solve,
    optimizer as _optimizer,
)
from simsopt_jax_adapters.geo import boozer_surface as _bsj


def _failed_eager_status(rhs):
    """A ``_LinearSolveStatus`` with ``success=False`` -- a zero solution makes
    the residual equal the RHS, so ``residual_relative == 1`` fails the gate.
    Stands in for the eager Hessian-LS probe failing on a loosely-converged
    surface."""
    return _optimizer._linear_solve_status(
        jnp.zeros_like(rhs),
        rhs,
        rhs,
        tol=1e-12,
        iterations=_optimizer._device_int32(0),
    )


def _wellconditioned_square(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((n, n)) + n * np.eye(n))


def test_reprobe_recovers_availability_when_flag_on(monkeypatch):
    """Flag on + eager failed + nonlinear converged + square J: the re-probe
    solves J^T x = grad via dense-LU, flips success to True, and returns the
    un-squared solution (verified against the dense oracle)."""
    monkeypatch.setattr(_linear_solve, "_EXACT_ADJOINT_DENSE_LU", True)
    jac = _wellconditioned_square(seed=1)
    n = jac.shape[0]
    grad = jnp.asarray(np.random.default_rng(2).standard_normal(n))
    solution, _status, success, backend = (
        _bsj._reprobe_adjoint_availability_with_dense_lu(
            jacobian=jac,
            grad=grad,
            tol=1e-11,
            eager_solution=jnp.zeros(n),
            eager_status=_failed_eager_status(grad),
            eager_success=False,
            nonlinear_success=True,
        )
    )
    assert success is True
    assert backend == "dense-lu-jacobian-transpose"
    expected = np.linalg.solve(np.asarray(jac).T, np.asarray(grad))
    np.testing.assert_allclose(
        np.asarray(solution), np.asarray(expected), rtol=1e-10, atol=1e-12
    )


def test_reprobe_inert_when_flag_off():
    """At the default (flag off) the re-probe is inert: it returns the eager
    result unchanged and does not run the dense-LU solver."""
    assert _linear_solve._EXACT_ADJOINT_DENSE_LU is False
    jac = _wellconditioned_square(seed=3)
    n = jac.shape[0]
    grad = jnp.ones(n)
    eager_solution = jnp.full(n, 7.0)
    solution, _status, success, backend = (
        _bsj._reprobe_adjoint_availability_with_dense_lu(
            jacobian=jac,
            grad=grad,
            tol=1e-11,
            eager_solution=eager_solution,
            eager_status=_failed_eager_status(grad),
            eager_success=False,
            nonlinear_success=True,
        )
    )
    assert success is False
    assert backend == "least-squares-hessian"
    np.testing.assert_array_equal(np.asarray(solution), np.asarray(eager_solution))


def test_reprobe_does_not_mask_nonlinear_failure(monkeypatch):
    """Even with the flag on and a recoverable Jacobian, the re-probe must NOT
    run when the nonlinear solve failed -- it never masks non-convergence."""
    monkeypatch.setattr(_linear_solve, "_EXACT_ADJOINT_DENSE_LU", True)
    jac = _wellconditioned_square(seed=4)
    n = jac.shape[0]
    grad = jnp.ones(n)
    eager_solution = jnp.zeros(n)
    solution, _status, success, backend = (
        _bsj._reprobe_adjoint_availability_with_dense_lu(
            jacobian=jac,
            grad=grad,
            tol=1e-11,
            eager_solution=eager_solution,
            eager_status=_failed_eager_status(grad),
            eager_success=False,
            nonlinear_success=False,
        )
    )
    assert success is False
    assert backend == "least-squares-hessian"


def test_reprobe_skips_nonsquare_jacobian(monkeypatch):
    """A non-square Jacobian has no square J^T to factor; the re-probe skips it
    and leaves the eager (failed) result in place."""
    monkeypatch.setattr(_linear_solve, "_EXACT_ADJOINT_DENSE_LU", True)
    jac = jnp.asarray(np.random.default_rng(5).standard_normal((10, 12)))
    grad = jnp.ones(10)
    _solution, _status, success, backend = (
        _bsj._reprobe_adjoint_availability_with_dense_lu(
            jacobian=jac,
            grad=grad,
            tol=1e-11,
            eager_solution=jnp.zeros(10),
            eager_status=_failed_eager_status(grad),
            eager_success=False,
            nonlinear_success=True,
        )
    )
    assert success is False
    assert backend == "least-squares-hessian"


def test_reprobe_inert_when_eager_already_succeeded(monkeypatch):
    """When the eager probe already succeeded the re-probe does nothing -- it is
    only a recovery path for the false-failure case."""
    monkeypatch.setattr(_linear_solve, "_EXACT_ADJOINT_DENSE_LU", True)
    jac = _wellconditioned_square(seed=6)
    n = jac.shape[0]
    eager_solution = jnp.full(n, 3.0)
    grad = jnp.ones(n)
    solution, _status, success, backend = (
        _bsj._reprobe_adjoint_availability_with_dense_lu(
            jacobian=jac,
            grad=grad,
            tol=1e-11,
            eager_solution=eager_solution,
            eager_status=_failed_eager_status(grad),
            eager_success=True,
            nonlinear_success=True,
        )
    )
    assert success is True
    assert backend == "least-squares-hessian"
    np.testing.assert_array_equal(np.asarray(solution), np.asarray(eager_solution))
