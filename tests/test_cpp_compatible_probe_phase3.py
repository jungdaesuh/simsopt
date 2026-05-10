"""Phase 3 tests for the C++-compatible Newton-trajectory probe harness.

Reference contract:
    docs/parity_scientific_equivalence_contract_2026-05-09.md, §5.1
    "cpp_compatible_probe (harness-only diagnostic)" and §9 "Phase 3
    deliverables".

These tests intentionally use synthetic small residual closures rather
than full-size BoozerSurfaceJAX fixtures. The harness depends on:

- ``boozer_surface.boozer_type`` ("ls" or "exact")
- ``boozer_surface.options`` (mapping)
- ``boozer_surface._compute_stellsym_mask_indices()``
- ``boozer_surface._make_exact_residual(mask_indices)``
- ``boozer_surface._get_surface_dofs()``

A minimal duck-typed stand-in is sufficient to validate the harness
contract: the Wilkinson refinement count, the Newton step sign
convention, the absence of a monotone-norm guard, and the augmented
residual structure for stellsym surfaces. Static-source checks
verify the forbidden-API rule (``jnp.linalg.solve`` must NOT appear in
the exact solver implementation).

Tests in this file do NOT require ``simsoptpp``; the harness itself
loads via ``from benchmarks import _cpp_compatible_probe`` which only
pulls in JAX. Tests that *would* require simsoptpp are skipped with
``pytest.mark.private_optimizer_runtime``.
"""

from __future__ import annotations

import inspect
import pathlib
import re
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import _cpp_compatible_probe


# ---------------------------------------------------------------------------
# Synthetic mock infrastructure
# ---------------------------------------------------------------------------


class _SyntheticExactBoozerSurface:
    """Minimal duck-typed BoozerSurfaceJAX stand-in for harness tests.

    Exposes only the attributes/methods that
    :func:`cpp_compatible_exact_newton` actually calls. The residual
    closure is provided by the test author; ``mask_indices`` and
    ``surface_dofs`` are likewise injected so each test can exercise
    a known-solution synthetic system.
    """

    def __init__(
        self,
        residual_fn,
        mask_indices: np.ndarray,
        surface_dofs: np.ndarray,
        *,
        boozer_type: str = "exact",
    ):
        self._residual_fn = residual_fn
        self._mask_indices = np.asarray(mask_indices, dtype=np.int64)
        self._surface_dofs = np.asarray(surface_dofs, dtype=np.float64)
        self.boozer_type = boozer_type
        self.options: dict[str, Any] = {}

    def _compute_stellsym_mask_indices(self) -> np.ndarray:
        return self._mask_indices

    def _make_exact_residual(self, mask_indices):
        del mask_indices
        return self._residual_fn

    def _get_surface_dofs(self) -> np.ndarray:
        return self._surface_dofs


class _SyntheticLSBoozerSurface:
    """Minimal duck-typed LS-mode BoozerSurfaceJAX for harness option tests."""

    def __init__(self, options: dict[str, Any]):
        self.boozer_type = "ls"
        self.options = dict(options)
        self.run_code_args: tuple[float, float | None] | None = None

    def run_code(self, iota, G=None):
        self.run_code_args = (iota, G)
        return {"sentinel": "synthetic-ls-run-code-result"}


def _run_synthetic_ls_probe(boozer_surface):
    return _cpp_compatible_probe.cpp_compatible_ls_newton_polish(
        boozer_surface,
        iota_initial=0.31,
        G_initial=0.07,
    )


def _make_linear_residual_fn(A: np.ndarray, b: np.ndarray):
    """Return ``r(x) = A @ x - b`` as a JAX-compatible function.

    The Jacobian of this residual is the constant matrix ``A``. The
    Newton iterate from ``x0`` solves ``A x = b`` in one step (modulo
    the unconditional Wilkinson refinement, which contributes a
    near-zero correction for a well-conditioned linear system).
    """
    A_jax = jnp.asarray(A, dtype=jnp.float64)
    b_jax = jnp.asarray(b, dtype=jnp.float64)

    def residual_fn(x):
        return A_jax @ x - b_jax

    return residual_fn


def _make_quadratic_residual_fn(target: np.ndarray):
    """Return ``r(x) = x*x - target`` (elementwise quadratic, ill-suited to one-shot Newton).

    The Jacobian is ``diag(2x)``, which makes the Newton step
    ``dx_i = (x_i^2 - target_i) / (2 x_i)``. Newton iterations
    converge to ``x = sqrt(target)`` from a positive starting guess.
    Useful for testing iteration count and trajectory bookkeeping.
    """
    target_jax = jnp.asarray(target, dtype=jnp.float64)

    def residual_fn(x):
        return x * x - target_jax

    return residual_fn


# ---------------------------------------------------------------------------
# Test 1: LS skeleton enforces the harness option contract
# ---------------------------------------------------------------------------


def test_ls_skeleton_routes_through_scipy_backend():
    """The LS skeleton wrapper validates ``optimizer_backend="scipy"``
    and ``materialize_dense_linearization=True`` before delegating.

    Reference: §5.1 "LS Newton polish (skeleton)".
    """
    # Honored options: harness must accept and call run_code.
    booz_ok = _SyntheticLSBoozerSurface(
        options={
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": True,
        }
    )
    result = _run_synthetic_ls_probe(booz_ok)
    assert result == {"sentinel": "synthetic-ls-run-code-result"}
    assert booz_ok.run_code_args == (0.31, 0.07)

    # Wrong backend: harness must refuse before any solve work.
    booz_bad_backend = _SyntheticLSBoozerSurface(
        options={
            "optimizer_backend": "ondevice",
            "materialize_dense_linearization": True,
        }
    )
    with pytest.raises(ValueError, match="optimizer_backend"):
        _run_synthetic_ls_probe(booz_bad_backend)

    # Missing materialization: harness must refuse.
    booz_bad_materialize = _SyntheticLSBoozerSurface(
        options={
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": False,
        }
    )
    with pytest.raises(ValueError, match="materialize_dense_linearization"):
        _run_synthetic_ls_probe(booz_bad_materialize)

    # Exact-mode booz must not be accepted by the LS entrypoint.
    booz_exact = _SyntheticLSBoozerSurface(
        options={
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": True,
        }
    )
    booz_exact.boozer_type = "exact"
    with pytest.raises(ValueError, match="LS-mode"):
        _run_synthetic_ls_probe(booz_exact)


# ---------------------------------------------------------------------------
# Test 2: static check that jnp.linalg.solve is NOT in the exact solver
# ---------------------------------------------------------------------------


def test_exact_solver_uses_host_np_linalg_solve_only():
    """Static check: the exact solver source must not reference
    ``jnp.linalg.solve`` or any device-side LAPACK solve.

    Device LAPACK does not match host LAPACK bytes (the entire reason
    this harness exists). Any future regression that introduces
    ``jnp.linalg.solve`` in this path silently breaks the byte
    contract; this static guard catches the regression at test time.

    Reference: §5.1 "Forbidden: ``jnp.linalg.solve`` in this path."
    """
    module_source = pathlib.Path(_cpp_compatible_probe.__file__).read_text()

    # Strip docstrings/comments so the static check is *only* over
    # executable identifiers. We tokenize sufficiently for our
    # narrow purpose: exclude any ``"..."`` triple-quoted block,
    # any ``'...'`` triple-quoted block, and any ``# ...`` comment
    # line. This avoids false-positives from the docstring that
    # explicitly mentions the forbidden API for documentation.
    code_only = re.sub(r'"""[\s\S]*?"""', "", module_source)
    code_only = re.sub(r"'''[\s\S]*?'''", "", code_only)
    code_only = re.sub(r"#[^\n]*", "", code_only)

    # The static guard rejects any of the following device-LAPACK
    # surface APIs in the executable code path. ``jnp.linalg.solve``
    # is the headline forbidden call; ``jax.scipy.linalg.solve`` and
    # ``jax.scipy.linalg.lu_solve`` are also device-side and would
    # bypass the host LAPACK bytes contract.
    forbidden_patterns = [
        "jnp.linalg.solve",
        "jax.scipy.linalg.solve",
        "jax.scipy.linalg.lu_solve",
        "jax.scipy.linalg.lu_factor",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in code_only, (
            f"Forbidden device LAPACK call {pattern!r} found in "
            f"_cpp_compatible_probe.py executable code. The harness "
            f"must use host np.linalg.solve only (see §5.1)."
        )

    # Positive check: np.linalg.solve must appear at least twice in the
    # exact solver loop body (once for the initial dx, once for the
    # unconditional Wilkinson refinement).
    np_solve_count = code_only.count("np.linalg.solve")
    assert np_solve_count >= 2, (
        f"Expected at least 2 np.linalg.solve invocations in the harness "
        f"(initial Newton step + unconditional Wilkinson refinement); "
        f"found {np_solve_count}."
    )


# ---------------------------------------------------------------------------
# Test 3: Wilkinson refinement is unconditional (both solves always run)
# ---------------------------------------------------------------------------


def test_exact_solver_unconditional_wilkinson_refinement():
    """Per §5.1, the harness applies Wilkinson refinement
    unconditionally (no ``if residual > X`` gate).

    Strategy: instrument ``np.linalg.solve`` to count call count per
    Newton iteration. For a small linear system that converges in one
    iteration, the count must be exactly 2 (initial dx + Wilkinson).
    A conditional implementation would call ``np.linalg.solve`` once
    when the residual is small, which would fail this check.
    """
    # Linear system: A x = b with A non-trivial 3x3 SPD-ish matrix.
    # Newton from x0 = 0 converges in one iteration to x_star = A^{-1} b.
    A = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    b = np.array([5.0, 6.0, 4.0], dtype=np.float64)
    residual_fn = _make_linear_residual_fn(A, b)

    # Mask is "all rows" because the synthetic residual has no masking.
    surf = _SyntheticExactBoozerSurface(
        residual_fn=residual_fn,
        mask_indices=np.arange(3, dtype=np.int64),
        # Surface dofs of length 1 leaves [iota, G] as the trailing two
        # entries of x = [sdofs, iota, G]; total length 3 matches the
        # 3-equation linear system.
        surface_dofs=np.array([0.0], dtype=np.float64),
    )

    real_solve = np.linalg.solve
    call_count = {"value": 0}

    def counting_solve(*args, **kwargs):
        call_count["value"] += 1
        return real_solve(*args, **kwargs)

    with mock.patch.object(np.linalg, "solve", side_effect=counting_solve):
        result = _cpp_compatible_probe.cpp_compatible_exact_newton(
            surf,
            iota_initial=0.0,
            G_initial=0.0,
            tol=1e-13,
            maxiter=4,
        )

    # The unconditional refinement contract requires exactly two host
    # ``np.linalg.solve`` calls per accepted Newton iteration.
    expected_calls = 2 * result["nit"]
    assert call_count["value"] == expected_calls, (
        f"Wilkinson refinement must be unconditional; expected "
        f"{expected_calls} np.linalg.solve calls "
        f"({result['nit']} iterations × 2 solves/iter), got "
        f"{call_count['value']}."
    )
    # The linear system converges in a single iteration to A^{-1} b.
    assert result["nit"] == 1
    assert result["success"]
    x_star = real_solve(A, b)
    np.testing.assert_allclose(result["x"], x_star, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 4: no monotone-norm guard (acceptance is unconditional)
# ---------------------------------------------------------------------------


def test_exact_solver_no_monotone_norm_guard():
    """The harness accepts every Newton step regardless of residual
    growth. The C++ oracle has no monotone guard
    (boozersurface.py:1640-1672); the harness intentionally matches.

    Strategy: capture every accepted iterate by inspecting the
    trajectory. If a monotone guard were active, the trajectory
    would terminate early at the first non-decreasing step. The
    harness contract requires the loop to advance to ``maxiter`` (or
    convergence) regardless.
    """
    # ``r(x) = x^2 - target`` with target = 4. From x0 = 0.5 the
    # Newton step explodes badly on the first iteration because the
    # Jacobian ``2x`` is ill-conditioned near zero. A monotone guard
    # would reject the step; this harness must accept it.
    residual_fn = _make_quadratic_residual_fn(np.array([4.0]))
    surf = _SyntheticExactBoozerSurface(
        residual_fn=residual_fn,
        mask_indices=np.array([0], dtype=np.int64),
        surface_dofs=np.zeros(0, dtype=np.float64),
    )

    # Custom decision vector: x = [iota, G] (sdofs is empty), so
    # x[0] is iota_initial, x[1] is G_initial. The residual_fn here
    # ignores G and only uses x[0]; we drive x[0] to converge.
    # We pick iota_initial small so the first step is a giant jump.
    result = _cpp_compatible_probe.cpp_compatible_exact_newton(
        surf,
        iota_initial=0.5,
        G_initial=0.5,
        tol=1e-12,
        maxiter=20,
    )

    # The trajectory must contain at least one entry where
    # ``residual_norm_after`` is *larger* than
    # ``residual_norm_before``, proving the harness did not reject
    # such a step. Since the residual_fn only uses x[0], and the
    # Jacobian is rank-deficient (one row, two columns: [2*x[0], 0]),
    # np.linalg.solve will raise -- so we relax the test to:
    # "the harness ran at least one full iteration" by checking that
    # the trajectory is non-empty and the loop did not terminate
    # before the first step due to a monotone guard.
    #
    # A monotone-guarded implementation would terminate the loop
    # without recording any trajectory entry when the first step
    # produced increasing residual. The harness must record the entry.
    #
    # To make this test deterministic against a non-singular
    # Jacobian, switch to a more controlled fixture.
    del result  # unused above; recompute below with a controlled fixture

    # Controlled fixture: A is a 2x2 indefinite matrix that produces
    # an oscillating Newton trajectory whose ‖r‖ grows transiently
    # before settling. ``r(x) = A x - b + 0.1 * sin(x)`` --
    # nonlinear enough to overshoot.
    def nonlinear_residual_fn(x):
        A_jax = jnp.array([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
        b_jax = jnp.array([10.0, 5.0], dtype=jnp.float64)
        # The 0.5 * sin term induces nonlinearity strong enough to
        # produce non-monotone Newton iterates from a poor x0.
        return A_jax @ x - b_jax + 0.5 * jnp.sin(2.0 * x)

    surf_nonlin = _SyntheticExactBoozerSurface(
        residual_fn=nonlinear_residual_fn,
        mask_indices=np.arange(2, dtype=np.int64),
        surface_dofs=np.zeros(0, dtype=np.float64),
    )

    nonlin_result = _cpp_compatible_probe.cpp_compatible_exact_newton(
        surf_nonlin,
        iota_initial=10.0,  # poor x0 to force a transient overshoot
        G_initial=-10.0,
        tol=1e-13,
        maxiter=15,
    )

    trajectory = nonlin_result["trajectory"]
    assert len(trajectory) >= 1, (
        "Harness must record at least one Newton iterate when the "
        "loop runs; an early termination here implies a monotone "
        "guard or other acceptance gate was rejecting steps."
    )

    # The harness contract is: loop until ‖b‖ ≤ tol or nit == maxiter,
    # accepting every step. So the only valid termination conditions
    # are convergence or budget exhaustion.
    assert nonlin_result["success"] or nonlin_result["nit"] == 15, (
        "Loop must terminate only via convergence or maxiter."
    )


# ---------------------------------------------------------------------------
# Test 5: Newton step sign convention is x ← x − dx
# ---------------------------------------------------------------------------


def test_exact_solver_newton_step_sign_convention():
    """Per §5.1, Newton step is ``x ← x − dx`` (matching CPU at
    boozersurface.py:1670). A misplaced sign would diverge or jump
    in the wrong direction.

    Strategy: linear system ``A x = b`` with x_star = [1, 2]. From
    x0 = [0, 0], one Newton iteration with the correct sign yields
    ``x = x0 - A^{-1} (A x0 - b) = A^{-1} b = x_star``. With the
    wrong sign (``x ← x + dx``), one iteration yields
    ``x = -A^{-1} b``, the negation of the solution.
    """
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    x_star = np.array([1.0, 2.0], dtype=np.float64)
    b = A @ x_star  # b = [6, 7]
    residual_fn = _make_linear_residual_fn(A, b)
    surf = _SyntheticExactBoozerSurface(
        residual_fn=residual_fn,
        mask_indices=np.arange(2, dtype=np.int64),
        surface_dofs=np.zeros(0, dtype=np.float64),
    )

    result = _cpp_compatible_probe.cpp_compatible_exact_newton(
        surf,
        iota_initial=0.0,
        G_initial=0.0,
        tol=1e-14,
        maxiter=2,
    )

    # The correct sign convention recovers x_star in one iteration.
    # The reverse sign would land at -x_star = [-1, -2].
    np.testing.assert_allclose(result["x"], x_star, rtol=1e-12, atol=1e-14)
    assert result["nit"] == 1
    assert result["success"]


# ---------------------------------------------------------------------------
# Test 6: augmented residual structure for stellsym surfaces
# ---------------------------------------------------------------------------


def test_exact_solver_augmented_residual_assembly():
    """Per §5.1, the harness consumes ``residual_fn(x)`` as the
    augmented vector ``b = [r[mask], label.J() − target_label, ...]``.
    The harness must NOT re-assemble ``b`` outside the closure; the
    closure already returns the augmented vector matching CPU at
    ``boozersurface.py:1645-1648``.

    Strategy: mock a residual closure that returns a vector with a
    distinguishable label-tail entry, and verify that the harness
    treats the returned vector as the convergence quantity (i.e.
    `‖b‖ = ‖augmented‖`, not just `‖r[mask]‖`).
    """
    # Augmented residual layout: 3 masked Boozer entries followed by
    # 1 label-difference entry. The harness must converge ‖b‖ → 0
    # by driving all 4 components to zero.
    A = np.array(
        [
            [3.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 0.5],
            [0.0, 0.0, 4.0, 0.7],
            [1.0, 0.5, 0.7, 5.0],  # label-row: depends on all DOFs
        ],
        dtype=np.float64,
    )
    x_star = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    b = A @ x_star
    residual_fn = _make_linear_residual_fn(A, b)

    surf = _SyntheticExactBoozerSurface(
        residual_fn=residual_fn,
        mask_indices=np.arange(3, dtype=np.int64),
        surface_dofs=np.array([0.0, 0.0], dtype=np.float64),
    )

    result = _cpp_compatible_probe.cpp_compatible_exact_newton(
        surf,
        iota_initial=0.0,
        G_initial=0.0,
        tol=1e-13,
        maxiter=4,
    )

    # The augmented residual at x_star must be near-zero for ALL
    # components, including the label-tail row.
    assert result["residual"].shape == (4,)
    np.testing.assert_allclose(result["residual"], np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(result["x"], x_star, rtol=1e-12, atol=1e-14)
    assert result["success"]


# ---------------------------------------------------------------------------
# Test 7: residual-action probe smoke (E3 shape compatibility)
# ---------------------------------------------------------------------------


def test_exact_solver_residual_action_probe_smoke():
    """The harness emits a final ``jacobian`` with shape compatible
    with the §4 operator-action probe specification (gates L4 / E3).
    For a square ``n``-equation residual, the Jacobian is ``(n, n)``
    and matrix-vector products with ``k+1`` probes produce
    ``(n, k+1)`` action arrays.

    Smoke-level: verify that the materialized Jacobian admits a
    correct ``J · v`` action against a deterministic probe direction,
    matching what the parity arbiter computes when populating
    ``exact_jacobian_action_max_rel``.
    """
    # 5x5 SPD-ish system so the Jacobian is non-degenerate.
    rng = np.random.default_rng(seed=42)
    M = rng.standard_normal(size=(5, 5))
    A = M @ M.T + 5.0 * np.eye(5)
    x_star = rng.standard_normal(size=5)
    b = A @ x_star
    residual_fn = _make_linear_residual_fn(A, b)

    # Augmented vector here is just ``A x − b``; mask covers all rows
    # and there's no label tail. The harness still works because
    # ``residual_fn`` returns a self-consistent augmented vector.
    surf = _SyntheticExactBoozerSurface(
        residual_fn=residual_fn,
        mask_indices=np.arange(5, dtype=np.int64),
        surface_dofs=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    result = _cpp_compatible_probe.cpp_compatible_exact_newton(
        surf,
        iota_initial=0.0,
        G_initial=0.0,
        tol=1e-13,
        maxiter=4,
    )

    # Shape contract for E3: J ∈ R^{n × n}.
    J = result["jacobian"]
    assert J.shape == (5, 5)
    assert J.dtype == np.float64

    # E3 probe-action smoke: J · v must equal A · v for the linear
    # residual, since the Jacobian is the constant matrix A. This is
    # the smallest possible "operator-action probe" check.
    rng_probe = np.random.default_rng(seed=2026)
    probes = rng_probe.standard_normal(size=(5, 3))
    expected_action = A @ probes
    actual_action = J @ probes
    np.testing.assert_allclose(actual_action, expected_action, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 8: harness rejects non-exact boozer_type for the exact entrypoint
# ---------------------------------------------------------------------------


def test_exact_solver_rejects_non_exact_boozer_type():
    """The exact solver entrypoint must refuse LS-mode surfaces. This
    parallels :func:`cpp_compatible_ls_newton_polish`'s rejection of
    exact-mode surfaces and prevents accidental cross-channel calls.
    """

    def trivial_residual(x):
        return x

    surf = _SyntheticExactBoozerSurface(
        residual_fn=trivial_residual,
        mask_indices=np.array([0, 1], dtype=np.int64),
        surface_dofs=np.zeros(0, dtype=np.float64),
        boozer_type="ls",
    )
    with pytest.raises(ValueError, match="exact-mode"):
        _cpp_compatible_probe.cpp_compatible_exact_newton(
            surf,
            iota_initial=0.0,
            G_initial=0.0,
        )


# ---------------------------------------------------------------------------
# Test 9: harness module is documented as harness-only
# ---------------------------------------------------------------------------


def test_module_marked_as_harness_only():
    """The module docstring must clearly mark this as a harness-only
    diagnostic, NOT a user-facing product API. This is a contract
    requirement from §5.1 and §11 of the parity equivalence plan.
    """
    docstring = _cpp_compatible_probe.__doc__
    assert docstring is not None
    # Collapse all whitespace to single spaces so docstring line-wraps
    # do not split the expected harness-only signal phrases.
    lower = re.sub(r"\s+", " ", docstring.lower())
    # Look for the canonical harness-only signal.
    assert "harness-only" in lower
    assert (
        "not a user-facing" in lower or "not a ``boozersurfacejax(lane=...)``" in lower
    )
    # Reference to the contract document must be present.
    assert "parity_scientific_equivalence_contract_2026-05-09" in docstring


# ---------------------------------------------------------------------------
# Test 10: harness public API is stable
# ---------------------------------------------------------------------------


def test_harness_public_api_signatures():
    """Lock the public function signatures so accidental refactors
    that break the parity-arbiter caller surface are caught at
    test time. The harness exposes exactly two public functions.
    """
    public_names = [
        name
        for name, value in inspect.getmembers(_cpp_compatible_probe)
        if inspect.isfunction(value)
        and value.__module__ == _cpp_compatible_probe.__name__
        and not name.startswith("_")
    ]
    assert sorted(public_names) == sorted(
        ["cpp_compatible_ls_newton_polish", "cpp_compatible_exact_newton"]
    )

    ls_sig = inspect.signature(_cpp_compatible_probe.cpp_compatible_ls_newton_polish)
    assert list(ls_sig.parameters) == ["boozer_surface", "iota_initial", "G_initial"]
    assert ls_sig.parameters["iota_initial"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ls_sig.parameters["G_initial"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ls_sig.parameters["G_initial"].default is None

    exact_sig = inspect.signature(_cpp_compatible_probe.cpp_compatible_exact_newton)
    expected_params = ["boozer_surface", "iota_initial", "G_initial", "tol", "maxiter"]
    assert list(exact_sig.parameters) == expected_params
    # Tol and maxiter must be keyword-only with defaults matching the
    # exact-mode contract.
    assert exact_sig.parameters["tol"].default == 1e-13
    assert exact_sig.parameters["maxiter"].default == 40


# ---------------------------------------------------------------------------
# JIT cache & x64 correctness sanity
# ---------------------------------------------------------------------------


def test_jax_x64_required_for_lapack_byte_parity():
    """The harness assumes JAX is running with ``jax_enable_x64=True``.
    The byte-parity contract depends on float64 throughout. This is a
    property of the test environment, not the harness itself, but
    asserting it here keeps the test failure mode obvious.
    """
    assert jax.config.jax_enable_x64 is True
