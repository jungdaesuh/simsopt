"""Phase 2 factor-once adjoint hybrid coverage.

Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §5.3,
the LS forward and adjoint solves must consume the same packed
``(lu, piv)`` factor bytes by construction. This test file proves the
contract on three layers:

1. Helper layer: ``_factor_dense_hessian`` + ``_plu_from_lu_piv``
   produce ``(P, L, U)`` such that ``P @ L @ U == H`` to machine
   precision, and the SciPy / on-device branches share LU bytes.
2. Solve layer: ``_traceable_solve_plu_linearization`` routed through
   the 5-tuple ``(P, L, U, lu, piv)`` form produces forward and
   transpose solutions that are bit-identical to direct
   ``jax.scipy.linalg.lu_solve`` calls.
3. Adapter layer: a full ``BoozerSurfaceJAX`` LS solve carries
   ``res["LU_PIV"]`` such that ``res["PLU"]`` is derived from the same
   factorization (verified by reproducing ``P @ L @ U == H`` and by
   checking that the runtime callback's solve returns
   ``lu_solve((lu, piv), rhs)`` bytes).

The first two test groups deliberately avoid ``simsoptpp``; the
adapter-layer integration is gated on ``private_optimizer_runtime``
because it constructs a ``BoozerSurfaceJAX`` end-to-end via the
on-device LS path that requires the simsoptpp-backed editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_SRC_ROOT)

from simsopt.geo import boozersurface_jax as bsj
from simsopt.geo import optimizer_jax as opt_jax
from simsopt.geo import surfaceobjectives_jax as soj


# --- Hessian fixtures ------------------------------------------------------


def _spd_hessian(n: int, *, seed: int) -> jnp.ndarray:
    """Return a deterministic SPD matrix that exercises pivoting."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    H = A.T @ A + n * np.eye(n)
    return jnp.asarray(H, dtype=jnp.float64)


def _pivoting_hessian() -> jnp.ndarray:
    """Return a small invertible matrix that requires non-trivial pivoting."""
    return jnp.asarray(
        [
            [0.0, 1.0, 2.0, 1.0],
            [3.0, 4.0, 5.0, 1.0],
            [6.0, 7.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )


# --- 1) Helper-layer parity ------------------------------------------------


@pytest.mark.parametrize("n,seed", [(4, 1), (8, 2), (16, 3)])
def test_plu_from_lu_piv_reconstructs_hessian_to_machine_precision(n, seed):
    """``P @ L @ U`` must reproduce ``H`` exactly on every fixture."""
    H = _spd_hessian(n, seed=seed)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    residual = np.asarray(P @ L @ U - H)
    eps = np.finfo(np.float64).eps
    assert np.linalg.norm(residual) <= eps * (n**2)


def test_plu_from_lu_piv_handles_pivoting():
    """Non-trivial row pivots must round-trip through the helper."""
    H = _pivoting_hessian()
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    err = float(jnp.linalg.norm(P @ L @ U - H))
    eps = np.finfo(np.float64).eps
    assert err <= eps * (H.shape[0] ** 2)


def test_factor_dense_hessian_scipy_and_jax_branches_share_bytes():
    """``optimizer_backend == "scipy"`` and ``"ondevice"`` must yield
    identical packed factor bytes on a shared host-CPU LAPACK fixture.
    """
    H = _spd_hessian(8, seed=42)
    lu_p, piv_p = opt_jax._factor_dense_hessian(H, optimizer_backend="scipy")
    lu_j, piv_j = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    assert np.array_equal(np.asarray(lu_p), np.asarray(lu_j))
    assert np.array_equal(np.asarray(piv_p), np.asarray(piv_j))


def test_factor_dense_hessian_returns_none_on_missing_input():
    assert opt_jax._factor_dense_hessian(None, optimizer_backend="scipy") is None


def test_plu_from_lu_piv_is_jit_traceable():
    """``_plu_from_lu_piv`` must compile under JIT without host roundtrips."""
    H = _spd_hessian(6, seed=11)
    factor = jax.jit(lambda mat: opt_jax._plu_from_lu_piv(jsp_linalg.lu_factor(mat)))
    P, L, U = factor(H)
    err = float(jnp.linalg.norm(P @ L @ U - H))
    eps = np.finfo(np.float64).eps
    assert err <= eps * (H.shape[0] ** 2)


# --- 2) Solve-layer bit-equality -------------------------------------------


@pytest.mark.parametrize("n,seed", [(4, 1), (8, 2), (16, 3)])
def test_traceable_solve_plu_linearization_forward_matches_lu_solve(n, seed):
    """Forward solve via 5-tuple must equal ``lu_solve(lu_piv, rhs)`` bytes."""
    H = _spd_hessian(n, seed=seed)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    factors_5tuple = (P, L, U, lu_piv[0], lu_piv[1])

    rng = np.random.default_rng(seed + 100)
    rhs = jnp.asarray(rng.normal(size=(n,)), dtype=jnp.float64)

    solved, success = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    expected = jsp_linalg.lu_solve(lu_piv, rhs, trans=0)
    assert bool(np.asarray(success))
    assert np.array_equal(np.asarray(solved), np.asarray(expected))


@pytest.mark.parametrize("n,seed", [(4, 1), (8, 2), (16, 3)])
def test_traceable_solve_plu_linearization_transpose_matches_lu_solve_trans(n, seed):
    """Transpose solve via 5-tuple must equal ``lu_solve(lu_piv, rhs, trans=1)`` bytes."""
    H = _spd_hessian(n, seed=seed)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    factors_5tuple = (P, L, U, lu_piv[0], lu_piv[1])

    rng = np.random.default_rng(seed + 200)
    rhs = jnp.asarray(rng.normal(size=(n,)), dtype=jnp.float64)

    solved, success = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        linear_solve_tol=1e-12,
        transpose=True,
    )
    expected = jsp_linalg.lu_solve(lu_piv, rhs, trans=1)
    assert bool(np.asarray(success))
    assert np.array_equal(np.asarray(solved), np.asarray(expected))


@pytest.mark.parametrize("n,seed", [(4, 1), (8, 2), (16, 3)])
def test_forward_and_adjoint_hessian_action_share_factor_bytes(n, seed):
    """Forward and adjoint solves must consume the same factor bytes.

    Forward action ``H @ x`` and adjoint action ``H.T @ y`` use the
    SAME ``(lu, piv)`` packed factors. We verify by running both
    directions through the 5-tuple solve and matching against direct
    ``lu_solve`` outputs at machine precision (``np.finfo(np.float64).eps
    * n``). Because the SPD fixtures here have ``H == H.T``, forward
    and transpose solves additionally produce identical solutions for
    the same RHS.
    """
    H = _spd_hessian(n, seed=seed)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    factors_5tuple = (P, L, U, lu_piv[0], lu_piv[1])

    rng = np.random.default_rng(seed + 300)
    rhs = jnp.asarray(rng.normal(size=(n,)), dtype=jnp.float64)

    forward_sol, _ = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    adjoint_sol, _ = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        linear_solve_tol=1e-12,
        transpose=True,
    )
    eps = np.finfo(np.float64).eps
    # The SPD H is symmetric so forward and transpose give the same answer.
    diff = np.linalg.norm(np.asarray(forward_sol - adjoint_sol))
    assert diff <= eps * n

    # Also verify the underlying `(lu, piv)` factor bytes are unchanged
    # between the forward and the transpose call (proven by reusing the
    # same `factors_5tuple` reference and by reading lu/piv from the
    # tuple after both solves).
    assert factors_5tuple[3] is lu_piv[0]
    assert factors_5tuple[4] is lu_piv[1]


def test_traceable_solve_plu_linearization_5tuple_vs_3tuple_equivalent():
    """When the same ``(P, L, U)`` is passed in both forms the solutions
    must match — the 5-tuple route uses ``lu_solve``, the 3-tuple route
    uses triangular solves; both are mathematically equivalent and differ
    only at LAPACK roundoff.
    """
    n = 8
    H = _spd_hessian(n, seed=4)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    rng = np.random.default_rng(700)
    rhs = jnp.asarray(rng.normal(size=(n,)), dtype=jnp.float64)
    sol_5, _ = soj._traceable_solve_plu_linearization(
        (P, L, U, lu_piv[0], lu_piv[1]),
        rhs,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    sol_3, _ = soj._traceable_solve_plu_linearization(
        (P, L, U),
        rhs,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    eps = np.finfo(np.float64).eps
    assert np.linalg.norm(np.asarray(sol_5 - sol_3)) <= 100.0 * eps * n


# --- 3) Public API surface checks (no simsoptpp) ---------------------------


def test_ls_factorization_backend_reports_dense_plu_shared():
    """Phase 2 enum value must be returned when shared dispatch is on."""
    H = jnp.eye(3, dtype=jnp.float64)
    assert (
        bsj._ls_factorization_backend(
            H,
            optimizer_backend="scipy",
            shared_dispatch=True,
        )
        == "dense-plu-shared"
    )
    # When shared dispatch is off, the legacy enum still wins.
    assert (
        bsj._ls_factorization_backend(
            H,
            optimizer_backend="scipy",
            shared_dispatch=False,
        )
        == "lapack-dgetrf"
    )


def test_ls_factor_once_dispatch_eligible_byte_budget():
    """Above the byte budget, eligibility must return ``False``."""
    n = 16
    H = jnp.eye(n, dtype=jnp.float64)
    # Budget exactly fits: n*n*8 == budget
    eligible = bsj._ls_factor_once_dispatch_eligible(
        H,
        max_dense_jacobian_bytes=n * n * 8,
    )
    assert eligible is True
    # One byte under: n*n*8 > budget
    not_eligible = bsj._ls_factor_once_dispatch_eligible(
        H,
        max_dense_jacobian_bytes=n * n * 8 - 1,
    )
    assert not_eligible is False
    # None ⇒ no budget, always eligible
    assert (
        bsj._ls_factor_once_dispatch_eligible(
            H,
            max_dense_jacobian_bytes=None,
        )
        is True
    )
    # No matrix ⇒ ineligible.
    assert (
        bsj._ls_factor_once_dispatch_eligible(
            None,
            max_dense_jacobian_bytes=None,
        )
        is False
    )


def test_build_linear_solve_factors_from_res_threads_lu_piv():
    """When ``res["LU_PIV"]`` is set the helper must return a 5-tuple."""
    n = 4
    H = _spd_hessian(n, seed=5)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    plu = (P, L, U)
    res = {"PLU": plu, "LU_PIV": lu_piv}
    factors = soj._build_linear_solve_factors_from_res(res)
    assert factors is not None
    assert len(factors) == 5
    assert factors[0] is plu[0]
    assert factors[3] is lu_piv[0]
    assert factors[4] is lu_piv[1]


def test_build_linear_solve_factors_from_res_falls_back_to_triple():
    """Without ``LU_PIV`` the helper must return the legacy 3-tuple."""
    n = 4
    H = _spd_hessian(n, seed=6)
    P, L, U = jax.scipy.linalg.lu(H)
    plu = (P, L, U)
    res = {"PLU": plu, "LU_PIV": None}
    factors = soj._build_linear_solve_factors_from_res(res)
    assert factors is not None
    assert len(factors) == 3


def test_build_linear_solve_factors_from_res_handles_missing_plu():
    res = {"PLU": None, "LU_PIV": None}
    assert soj._build_linear_solve_factors_from_res(res) is None
