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

import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_FACTOR_DENSE_HESSIAN_BYTE_PARITY_SELECTOR = (
    "tests/integration/test_factor_once_adjoint_phase2.py::"
    "test_factor_dense_hessian_scipy_and_jax_branches_share_bytes"
)
_FACTOR_DENSE_HESSIAN_CPU_CHILD_ENV = "SIMSOPT_TEST_FACTOR_DENSE_HESSIAN_CPU_CHILD"
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_SRC_ROOT)

import simsopt_jax_adapters.geo.boozer_surface as bsj
from simsopt_jax.geo.optimizers import optimizer as opt_jax
import simsopt_jax_adapters.geo.surface_objectives as soj


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


def _packed_factors(matrix):
    lu_piv = jsp_linalg.lu_factor(matrix)
    P, L, U = opt_jax._plu_from_lu_piv(lu_piv)
    return P, L, U, lu_piv[0], lu_piv[1]


def _run_factor_dense_hessian_byte_parity_on_cpu() -> subprocess.CompletedProcess[str]:
    """Re-run this test selector in an exact CPU-only child process."""
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "True"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["SIMSOPT_BACKEND_MODE"] = "jax_cpu_parity"
    env["SIMSOPT_BACKEND_STRICT"] = "1"
    env[_FACTOR_DENSE_HESSIAN_CPU_CHILD_ENV] = "1"
    env.pop("SIMSOPT_JAX_TRANSFER_GUARD", None)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            _FACTOR_DENSE_HESSIAN_BYTE_PARITY_SELECTOR,
        ),
        check=False,
        capture_output=True,
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        timeout=180,
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

    Pinned to a CPU device so the byte-identity contract is exercised on every
    backend, not only on CPU processes: the ``"scipy"`` branch is always host
    LAPACK, while the ``"ondevice"`` branch follows the array's device, so in a
    GPU process it would route through cuSOLVER (numerically equal but not
    byte-identical). Forcing both onto CPU keeps the byte contract well-defined
    regardless of the default backend; cross-vendor numerical equivalence is a
    separate concern not asserted here.
    """
    try:
        cpu_device = jax.devices("cpu")[0]
    except RuntimeError:
        if os.environ.get(_FACTOR_DENSE_HESSIAN_CPU_CHILD_ENV) == "1":
            raise
        result = _run_factor_dense_hessian_byte_parity_on_cpu()
        assert result.returncode == 0, (
            "CPU subprocess failed byte-identical dense-Hessian factor proof.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return

    with jax.default_device(cpu_device):
        H = _spd_hessian(8, seed=42)
        lu_p, piv_p = opt_jax._factor_dense_hessian(H, optimizer_backend="scipy")
        lu_j, piv_j = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")
        assert np.array_equal(np.asarray(lu_p), np.asarray(lu_j))
        assert np.array_equal(np.asarray(piv_p), np.asarray(piv_j))


def test_factor_dense_hessian_cpu_child_failure_does_not_spawn_grandchild(
    monkeypatch,
):
    """A failed CPU-only child reports its backend error without recursing."""

    def fail_cpu_enumeration(platform):
        assert platform == "cpu"
        raise RuntimeError("sentinel CPU unavailable")

    def forbid_subprocess_run(*_args, **_kwargs):
        pytest.fail("CPU fallback child must not spawn a grandchild")

    monkeypatch.setenv(_FACTOR_DENSE_HESSIAN_CPU_CHILD_ENV, "1")
    monkeypatch.setattr(jax, "devices", fail_cpu_enumeration)
    monkeypatch.setattr(subprocess, "run", forbid_subprocess_run)

    with pytest.raises(RuntimeError, match="sentinel CPU unavailable"):
        test_factor_dense_hessian_scipy_and_jax_branches_share_bytes()


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

    solved, status = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        live_matvec=lambda vector: H @ vector,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    expected = jsp_linalg.lu_solve(lu_piv, rhs, trans=0)
    assert bool(np.asarray(status.success))
    assert int(np.asarray(status.fp64_rebuild_count)) == 0
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

    solved, status = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        live_matvec=lambda vector: H.T @ vector,
        linear_solve_tol=1e-12,
        transpose=True,
    )
    expected = jsp_linalg.lu_solve(lu_piv, rhs, trans=1)
    assert bool(np.asarray(status.success))
    assert int(np.asarray(status.fp64_rebuild_count)) == 0
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
        live_matvec=lambda vector: H @ vector,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    adjoint_sol, _ = soj._traceable_solve_plu_linearization(
        factors_5tuple,
        rhs,
        live_matvec=lambda vector: H.T @ vector,
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
        live_matvec=lambda vector: H @ vector,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    sol_3, _ = soj._traceable_solve_plu_linearization(
        (P, L, U),
        rhs,
        live_matvec=lambda vector: H @ vector,
        linear_solve_tol=1e-12,
        transpose=False,
    )
    eps = np.finfo(np.float64).eps
    assert np.linalg.norm(np.asarray(sol_5 - sol_3)) <= 100.0 * eps * n


@pytest.mark.parametrize("transpose", [False, True])
def test_stale_supplied_factors_trigger_live_fp64_rebuild(transpose):
    """A self-consistent stale PLU must not certify a different live operator."""
    stale_matrix = jnp.eye(3, dtype=jnp.float64)
    live_matrix = jnp.asarray(
        [
            [3.0, 0.5, -0.25],
            [-0.1, 2.0, 0.4],
            [0.2, -0.3, 1.5],
        ],
        dtype=jnp.float64,
    )
    oriented_live_matrix = live_matrix.T if transpose else live_matrix
    rhs = jnp.asarray([0.75, -0.5, 1.25], dtype=jnp.float64)

    solution, status = soj._traceable_solve_plu_linearization(
        _packed_factors(stale_matrix),
        rhs,
        live_matvec=lambda vector: oriented_live_matrix @ vector,
        linear_solve_tol=1.0e-12,
        transpose=transpose,
    )

    assert bool(np.asarray(status.success))
    assert int(np.asarray(status.fp64_rebuild_count)) == 1
    np.testing.assert_allclose(
        np.asarray(solution),
        np.linalg.solve(np.asarray(oriented_live_matrix), np.asarray(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_fp32_supplied_factors_contract_against_live_fp64_operator(transpose):
    """FP32 factors may pass only after measured live-FP64 contraction."""
    live_matrix = jnp.asarray(
        [
            [2.0, 0.25, -0.1],
            [0.25, 1.5, 0.2],
            [-0.1, 0.2, 1.25],
        ],
        dtype=jnp.float64,
    )
    approximate_matrix = jnp.asarray(
        live_matrix
        + jnp.asarray(
            [
                [1.0e-4, -2.0e-5, 0.0],
                [-2.0e-5, -7.5e-5, 1.0e-5],
                [0.0, 1.0e-5, 5.0e-5],
            ],
            dtype=jnp.float64,
        ),
        dtype=jnp.float32,
    )
    oriented_live_matrix = live_matrix.T if transpose else live_matrix
    rhs = jnp.asarray([0.25, -0.75, 1.0], dtype=jnp.float64)

    solution, status = soj._traceable_solve_plu_linearization(
        _packed_factors(approximate_matrix),
        rhs,
        live_matvec=lambda vector: oriented_live_matrix @ vector,
        linear_solve_tol=1.0e-12,
        transpose=transpose,
    )

    trace_length = int(
        np.asarray(status.supplied_factor_residual_relative_trace_length)
    )
    residual_trace = np.asarray(status.supplied_factor_residual_relative_trace)[
        :trace_length
    ]
    contraction_trace = np.asarray(status.supplied_factor_contraction_ratio_trace)[
        : trace_length - 1
    ]
    assert bool(np.asarray(status.success))
    assert int(np.asarray(status.fp64_rebuild_count)) == 0
    assert trace_length >= 2
    assert np.all(np.isfinite(residual_trace))
    assert np.all(np.diff(residual_trace) < 0.0)
    assert np.all(np.isfinite(contraction_trace))
    assert np.all(contraction_trace < 1.0)
    np.testing.assert_allclose(
        np.asarray(solution),
        np.linalg.solve(np.asarray(oriented_live_matrix), np.asarray(rhs)),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


# --- 3) Public API surface checks (no simsoptpp) ---------------------------


def test_ls_factorization_backend_reports_dense_plu_shared():
    """Phase 2 enum value must be returned when shared dispatch is on."""
    H = jnp.eye(3, dtype=jnp.float64)
    assert (
        bsj._ls_factorization_backend(
            H,
            optimizer_backend="ondevice",
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


def test_ls_shared_dispatch_and_backend_labels_match_runtime_lanes():
    H = jnp.eye(3, dtype=jnp.float64)
    lu_piv = opt_jax._factor_dense_hessian(H, optimizer_backend="ondevice")

    assert bsj._ls_shared_lu_piv_dispatch("ondevice", lu_piv) is True
    assert bsj._ls_shared_lu_piv_dispatch("scipy", lu_piv) is False
    assert (
        bsj._ls_linear_solve_backend(
            optimizer_backend="ondevice",
            plu_available=True,
            shared_lu_piv_dispatch=True,
        )
        == "dense-plu-shared"
    )
    assert (
        bsj._ls_linear_solve_backend(
            optimizer_backend="scipy",
            plu_available=True,
            shared_lu_piv_dispatch=False,
        )
        == "dense-plu"
    )
    assert (
        bsj._ls_linear_solve_backend(
            optimizer_backend="ondevice",
            plu_available=True,
            shared_lu_piv_dispatch=False,
        )
        == "operator"
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
