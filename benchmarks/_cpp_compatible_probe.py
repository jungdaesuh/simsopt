"""Harness-only C++-compatible Newton trajectory probe (Phase 3).

Reference contract:
    docs/parity_scientific_equivalence_contract_2026-05-09.md, §5.1
    "cpp_compatible_probe (harness-only diagnostic)".

WARNING: this module is a **harness-only diagnostic**. It is NOT a
``BoozerSurfaceJAX(lane=...)`` constructor parameter and NOT a
user-facing product API. Production single-stage code paths must not
import these helpers. The harness exists to materialize a dense
host-resident Newton trajectory for byte-comparable parity with the C++
oracle in ``boozersurface.py`` (LS Newton polish in
``minimize_boozer_penalty_constraints_newton`` and the BoozerExact Newton
in ``solve_residual_equation_exactly_newton``).

Two channels are exposed:

1. :func:`cpp_compatible_ls_newton_polish` — thin convenience wrapper
   around the existing ``optimizer_backend="scipy"`` LS pathway. It
   does **not** introduce a new code path; it simply pins the option
   contract that makes the LS skeleton byte-comparable to the C++
   oracle within LAPACK pivot tie-breaks (``optimizer_backend="scipy"``,
   ``materialize_dense_linearization=True`` so ``newton_polish``
   receives ``dense_newton_steps=materialize_hessian``; mirror-upper
   symmetrization is the default in ``_materialize_dense_hessian``).

2. :func:`cpp_compatible_exact_newton` — a NEW dense host-resident
   exact Newton solver that lives entirely in this harness. The
   normalizer ``_normalize_solver_options`` in ``boozersurface_jax.py``
   drops ``optimizer_backend`` from the user-visible exact path; this
   harness **does not** modify that normalizer and **does not** expose
   the dense path to the user constructor. The harness reuses the
   public residual closure ``BoozerSurfaceJAX._make_exact_residual``
   (which already returns the augmented residual ``b = [r[mask],
   label.J() − target_label, …]``, matching the augmented residual the
   C++ ``solve_residual_equation_exactly_newton`` assembles),
   materializes the Jacobian via ``jax.jit(jax.jacobian(r))(x)`` to a
   host ``np.ndarray``, and runs the C++-equivalent Newton iteration:

   - ``dx = np.linalg.solve(J, b)``
   - **unconditional** Wilkinson refinement
     ``dx += np.linalg.solve(J, b − J @ dx)`` (matches the C++ oracle)
   - step ``x ← x − dx`` (matches the C++ oracle)
   - **no** monotone-norm guard (matches C++ behavior; the JAX
     production exact path adds a guard, the harness intentionally does
     not)

   Convergence is checked on the augmented residual norm ``‖b‖``
   exactly as the C++ oracle does, **not** on the raw masked Boozer
   residual.

   ``jnp.linalg.solve`` is **forbidden** in this path: device LAPACK
   does not match host LAPACK bytes, which is the entire reason this
   harness exists. All linear algebra below uses ``np.linalg.solve``
   on host ``np.ndarray`` operands. See the failing-test
   ``test_exact_solver_uses_host_np_linalg_solve_only`` for the static
   check.

The harness is invoked by direct import from the parity benchmark, e.g.::

    from benchmarks._cpp_compatible_probe import cpp_compatible_exact_newton
    result = cpp_compatible_exact_newton(boozer_surface, ...)

Nothing in the production single-stage pipeline calls into this module.
The exact solver returns its trajectory metadata (final ``x``, residual,
Jacobian, iteration count, success flag) so the parity arbiter can
populate the Exact gates E1–E6 reporting fields per §2.2 of the
contract.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Harness-only contract: this module must not be reachable from
# ``simsopt.geo.boozersurface_jax``. Import order is one-directional --
# benchmarks depend on simsopt, never the other way around. Any future code
# change that imports ``benchmarks._cpp_compatible_probe`` from inside the
# ``simsopt`` package is a violation of §5.1 and must be reverted.
# ---------------------------------------------------------------------------


# The harness-only LS skeleton pins these options when wrapping
# ``BoozerSurfaceJAX.run_code()``. They reproduce the C++ Newton polish
# byte semantics within LAPACK pivot tie-breaks. See
# ``_run_newton_polish_for_method`` in ``boozersurface_jax.py`` for the
# dispatcher that consumes these options and ``_solve_dense_newton_step``
# in ``optimizer_jax.py`` for the host ``np.linalg.solve`` site.
_LS_HARNESS_REQUIRED_OPTIONS: Mapping[str, Any] = {
    "optimizer_backend": "scipy",
    "materialize_dense_linearization": True,
}


def _validate_ls_harness_options(boozer_surface) -> None:
    """Raise ``ValueError`` unless the LS harness option pins are honored.

    The C++ byte-comparable LS skeleton requires:
    - ``optimizer_backend="scipy"`` so the SciPy-backed Newton polish
      path is selected (no ondevice routing);
    - ``materialize_dense_linearization=True`` so
      ``_run_newton_polish_for_method`` passes ``dense_newton_steps=True``
      to ``newton_polish``, which forces the host ``np.linalg.solve``
      site in ``_solve_dense_newton_step``.

    The harness must not silently mutate ``boozer_surface.options``;
    callers are responsible for setting the contract before invoking
    the probe. This validator surfaces drift early so the harness
    cannot return a result that lacks bit-comparable C++ semantics.
    """
    options = boozer_surface.options
    for key, expected in _LS_HARNESS_REQUIRED_OPTIONS.items():
        actual = options.get(key)
        if actual != expected:
            raise ValueError(
                f"cpp_compatible_ls_newton_polish requires option {key!r} "
                f"to be {expected!r} for byte-comparable C++ semantics; "
                f"got {actual!r}. Set this on the BoozerSurfaceJAX "
                f"constructor before invoking the harness."
            )


def cpp_compatible_ls_newton_polish(
    boozer_surface,
    *,
    iota_initial: float,
    G_initial: float | None = None,
):
    """Run the C++-compatible LS Newton-polish skeleton on a BoozerSurfaceJAX.

    Reference: ``docs/parity_scientific_equivalence_contract_2026-05-09.md``
    §5.1 "LS Newton polish (skeleton)".

    This is a thin wrapper that forwards to the existing
    ``optimizer_backend="scipy"`` LS pathway. It does NOT introduce a
    new dense code path -- the host ``np.linalg.solve`` Newton step
    site in ``_solve_dense_newton_step`` is already reached by the
    upstream dispatcher when ``dense_newton_steps=True`` is set, which
    happens when ``materialize_dense_linearization=True`` (the dispatcher
    forwards it as ``dense_newton_steps=materialize_hessian``).

    The wrapper exists for two reasons:

    1. To document the harness-required option contract in one place.
    2. To validate that the option contract is honored before invoking
       the solve, so a misconfigured caller cannot silently get a
       non-byte-comparable result through this entrypoint.

    Args:
        boozer_surface: A ``BoozerSurfaceJAX`` constructed with
            ``constraint_weight is not None`` (LS mode), with the
            constructor option contract honoring
            ``optimizer_backend="scipy"`` and
            ``materialize_dense_linearization=True``.
        iota_initial: Initial guess for the rotational transform. This
            is forwarded to ``BoozerSurfaceJAX.run_code(iota, G=...)``.
        G_initial: Optional initial guess for ``G``. ``None`` preserves
            the standard ``run_code`` contract where ``G`` is computed
            from fixed coil currents.

    Returns:
        The standard LS result dict from
        ``BoozerSurfaceJAX.run_code()``. Mirror-upper Hessian
        symmetrization is the default in ``_materialize_dense_hessian``
        and is requested explicitly with ``symmetrize=True`` at the
        dense-newton call site, so the materialized ``hessian`` field is
        bit-symmetric.
    """
    if boozer_surface.boozer_type != "ls":
        raise ValueError(
            "cpp_compatible_ls_newton_polish requires an LS-mode "
            "BoozerSurfaceJAX (constraint_weight is not None); "
            f"got boozer_type={boozer_surface.boozer_type!r}."
        )
    _validate_ls_harness_options(boozer_surface)
    G_value = None if G_initial is None else float(G_initial)
    return boozer_surface.run_code(
        float(iota_initial),
        G=G_value,
    )


def _materialize_jacobian_host(residual_fn, x_jax) -> np.ndarray:
    """Return ``J(x)`` as a host ``np.ndarray`` via ``jax.jacobian``.

    The harness uses ``jax.jit(jax.jacobian(residual_fn))(x)`` exactly
    as specified in §5.1 of the contract. The ``np.asarray`` host pull
    happens *outside* any ``jit``-traced region (it operates on the
    materialized result of the JIT call), satisfying the
    "no ``np.asarray`` on traced values" rule from §6.

    Float64 dtype is enforced on the host side because the host
    LAPACK byte-comparable semantics depend on it; the JAX runtime is
    expected to be running with ``jax_enable_x64=True``.
    """
    jacobian_fn_jit = jax.jit(jax.jacobian(residual_fn))
    j_jax = jacobian_fn_jit(x_jax)
    return np.asarray(j_jax, dtype=np.float64)


def _evaluate_residual_host(residual_fn, x_jax) -> np.ndarray:
    """Return ``r(x)`` as a host ``np.ndarray`` (float64).

    The residual closure is already ``jax.jit``-eligible; we evaluate
    on the device, then pull to host outside any traced region.
    """
    r_jax = residual_fn(x_jax)
    return np.asarray(r_jax, dtype=np.float64)


def cpp_compatible_exact_newton(
    boozer_surface,
    *,
    iota_initial: float,
    G_initial: float,
    tol: float = 1e-13,
    maxiter: int = 40,
):
    """Dense host-resident exact Newton solver for the BoozerExact path.

    Reference: ``docs/parity_scientific_equivalence_contract_2026-05-09.md``
    §5.1 "BoozerExact (NEW dense host-resident solver)".

    THIS IS A HARNESS-ONLY DIAGNOSTIC. The production exact path uses
    operator-backed GMRES inside ``newton_exact`` in
    ``boozersurface_jax.py``; this harness intentionally bypasses that
    for byte-comparable parity with the C++ oracle
    ``solve_residual_equation_exactly_newton`` in ``boozersurface.py``.
    Nothing in the production single-stage pipeline calls this function.

    The exact normalizer ``_normalize_solver_options`` in
    ``boozersurface_jax.py`` drops ``optimizer_backend`` from the
    user-visible exact path; this harness does NOT modify that
    normalizer. Instead, the harness builds the residual closure via the
    public ``BoozerSurfaceJAX._make_exact_residual`` factory (which
    already returns the augmented residual matching the C++
    ``b = [r[mask], label.J() − target_label, ...]`` assembly), then
    runs the C++-equivalent Newton iteration on host ``np.ndarray``
    factors.

    C++-equivalent Newton iteration (matches the C++
    ``solve_residual_equation_exactly_newton`` body):

    - augmented residual ``b = residual_fn(x)``  (already includes the
      label term and, for non-stellsym, the axis-z constraint)
    - ``dx = np.linalg.solve(J, b)``
    - **unconditional** Wilkinson refinement (one step):
      ``dx += np.linalg.solve(J, b − J @ dx)``  (matches the C++ oracle)
    - step ``x ← x − dx``  (matches the C++ oracle)
    - **no** monotone-norm guard
    - convergence check: ``‖b‖_2 ≤ tol``  (matches the C++ oracle)

    ``jnp.linalg.solve`` is forbidden in this path. All factor work
    happens on host ``np.ndarray`` operands using ``np.linalg.solve``,
    because device LAPACK does not match host LAPACK bytes.

    Args:
        boozer_surface: A ``BoozerSurfaceJAX`` constructed with
            ``constraint_weight is None`` (exact mode). Used only to
            access the residual factory
            ``_make_exact_residual(mask_indices)`` and the static
            mask indices ``_compute_stellsym_mask_indices()``.
            ``boozer_surface.run_code()`` is **not** called.
        iota_initial: Initial guess for the rotational transform.
        G_initial: Initial guess for ``G``. The harness does NOT
            recompute ``G`` from coil currents; the caller must
            supply the byte-equivalent value the C++ oracle uses.
        tol: ``‖b‖_2`` convergence tolerance. Default ``1e-13``
            matches ``_DEFAULT_OPTIONS_EXACT["newton_tol"]`` in
            ``boozersurface_jax.py``.
        maxiter: Maximum Newton iterations. Default ``40`` matches
            ``_DEFAULT_OPTIONS_EXACT["newton_maxiter"]``.

    Returns:
        Dict with the keys:

        - ``"x"``: final decision vector ``[sdofs, iota, G]`` as host
          ``np.ndarray`` (float64).
        - ``"sdofs"``, ``"iota"``, ``"G"``: split components of ``x``.
        - ``"residual"``: final augmented residual ``b(x_final)`` as
          host ``np.ndarray``.
        - ``"jacobian"``: final augmented Jacobian ``J(x_final)`` as
          host ``np.ndarray``, materialized via
          ``jax.jit(jax.jacobian(...))``.
        - ``"residual_norm"``: ``‖b(x_final)‖_2``.
        - ``"nit"``: completed Newton iteration count.
        - ``"success"``: ``True`` iff ``residual_norm <= tol``.
        - ``"converged"``: alias of ``"success"`` for parity-arbiter
          consumers.
        - ``"trajectory"``: list of per-iteration dicts with keys
          ``{"iter", "residual_norm_before", "dx_norm",
            "wilkinson_correction_norm", "residual_norm_after"}`` so
          the parity arbiter can populate Exact gates E2 (residual
          parity) and E5 (refinement-correction parity) per-iter.

        ``"linear_solve_backend"`` is reported as ``"lapack-dgetrf"``
        (the host ``np.linalg.solve`` LAPACK backend); ``"jacobian_materialized"``
        is always ``True`` because the harness owns the dense path.
    """
    if boozer_surface.boozer_type != "exact":
        raise ValueError(
            "cpp_compatible_exact_newton requires an exact-mode "
            "BoozerSurfaceJAX (constraint_weight is None); "
            f"got boozer_type={boozer_surface.boozer_type!r}."
        )
    if not isinstance(maxiter, int) or maxiter < 0:
        raise ValueError(f"maxiter must be a non-negative int; got {maxiter!r}.")
    tol_value = float(tol)
    if not (tol_value >= 0.0):
        raise ValueError(f"tol must be non-negative; got {tol!r}.")

    mask_indices = boozer_surface._compute_stellsym_mask_indices()
    residual_fn = boozer_surface._make_exact_residual(mask_indices)

    # The decision vector x = [sdofs, iota, G] follows the
    # BoozerSurfaceJAX exact-mode layout exactly. The harness materializes
    # x as a host np.ndarray to keep all linear algebra on host.
    sdofs = np.asarray(boozer_surface._get_surface_dofs(), dtype=np.float64)
    x_host = np.concatenate(
        (sdofs, np.asarray([float(iota_initial), float(G_initial)], dtype=np.float64))
    )

    # The residual closure expects a JAX-compatible array; we keep a
    # device-resident view of x synchronized with the host trajectory
    # so the JIT cache reuses the same trace across iterations.
    trajectory: list[dict[str, float | int]] = []
    nit = 0
    converged = False

    # Initial residual evaluation. This populates b_host so the
    # convergence check can fire even when maxiter == 0.
    x_jax = jnp.asarray(x_host, dtype=jnp.float64)
    b_host = _evaluate_residual_host(residual_fn, x_jax)
    b_norm = float(np.linalg.norm(b_host))

    while nit < maxiter and b_norm > tol_value:
        # Materialize J(x) on host. ``jax.jit(jax.jacobian(...))``
        # matches §5.1 verbatim. ``np.asarray`` happens outside the
        # JIT region by virtue of operating on the *result* of the
        # JIT call, so it does not violate the §6 "no host
        # roundtrips inside traced regions" rule.
        J_host = _materialize_jacobian_host(residual_fn, x_jax)

        # Wilkinson refinement matches CPU oracle:
        #     dx = np.linalg.solve(J, b)
        #     dx += np.linalg.solve(J, b - J @ dx)
        # The refinement is UNCONDITIONAL: there is no `if norm > X`
        # gate. Both np.linalg.solve calls always run regardless of
        # the residual size, exactly matching the C++ behavior.
        dx_initial = np.linalg.solve(J_host, b_host)
        wilkinson_rhs = b_host - J_host @ dx_initial
        wilkinson_correction = np.linalg.solve(J_host, wilkinson_rhs)
        dx_refined = dx_initial + wilkinson_correction

        dx_norm = float(np.linalg.norm(dx_refined))
        wilkinson_norm = float(np.linalg.norm(wilkinson_correction))
        b_norm_before = b_norm

        # CPU step convention: x -= dx. The harness intentionally does
        # NOT apply a monotone-norm guard; the C++ oracle accepts the
        # step unconditionally.
        x_host = x_host - dx_refined
        x_jax = jnp.asarray(x_host, dtype=jnp.float64)

        # Recompute b at the new iterate so the loop guard reflects
        # the current state. The convergence check uses ‖b‖, not
        # ‖raw masked residual‖, matching the C++ oracle.
        b_host = _evaluate_residual_host(residual_fn, x_jax)
        b_norm = float(np.linalg.norm(b_host))

        nit += 1
        trajectory.append(
            {
                "iter": nit,
                "residual_norm_before": b_norm_before,
                "dx_norm": dx_norm,
                "wilkinson_correction_norm": wilkinson_norm,
                "residual_norm_after": b_norm,
            }
        )

    converged = b_norm <= tol_value

    # Final Jacobian materialization for downstream reporting (E3
    # operator-action probes, E7 adjoint solve residual). When the
    # loop ran zero iterations, this still emits the J at the initial
    # iterate, which is what the parity arbiter expects.
    J_final = _materialize_jacobian_host(residual_fn, x_jax)

    sdofs_final = x_host[:-2].copy()
    iota_final = float(x_host[-2])
    G_final = float(x_host[-1])

    return {
        "x": x_host.copy(),
        "sdofs": sdofs_final,
        "iota": iota_final,
        "G": G_final,
        "residual": b_host.copy(),
        "jacobian": J_final,
        "residual_norm": b_norm,
        "nit": int(nit),
        "success": bool(converged),
        "converged": bool(converged),
        "trajectory": trajectory,
        "linear_solve_backend": "lapack-dgetrf",
        "jacobian_materialized": True,
    }
