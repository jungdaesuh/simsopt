# Nested-LS Newton parity canary (2026-08-20)

**Verdict: canary-1 code lock, not an F3 speed claim.** JAX and native C++
solve the same nested Boozer LS Newton problem on frozen coils. F3's 7.70×
is LS-flat GPU vs flat-native L-BFGS-B and does not transfer.

## Contract

Inner problem: `BoozerSurface.minimize_boozer_penalty_constraints_newton`
with `constraint_weight=1.0`, free `G`, `weight_inv_modB=True`,
`stab=1e-4`, `tol=1e-13` on `||∇J_LS||_2`, Volume label. State is
`[s, ι, G]`. Native `result['residual']` **is** `∇J_LS`. JAX stores `∇J_LS`
in `jacobian` and the long residual in `residual`; those two objects must
not be compared.

This is **not** banana `run_code` (BFGS then Newton, `stab=0`, `tol=1e-11`)
and **not** exact Newton on 21×21.

## What this receipt covers

`src/simsopt_jax_adapters/geo/nested_ls_newton_parity.py` and
`tests/geo/test_nested_ls_newton_parity.py` lock:

1. `J_LS` and `∇J_LS` at a packed state (`cpu_ordered` JAX reduction).
2. On-manifold Newton polish after a **successful** native LBFGS seed.
3. At least one Newton step from a `+1e-3` ι perturbation, still to
   `||∇J_LS||_2 ≤ 1e-13` on both lanes.
4. Native Newton `residual is jacobian` (length `n_s+2`); JAX Newton
   `residual` is the long LS vector, not `∇J_LS`.
5. Public JAX LS Newton persist: C++ `_boozer_iterate_is_persistable`
   (`success` or finite `‖∇J_LS‖₂` not worse than start) plus a JAX
   finite-`x`/finite-`∇J` conjunct (a `success=True` polish with a
   non-finite iterate rolls back). An exact no-move iterate stays
   persisted so a second autodiff stream cannot drop Hessian on ULP
   noise. Rollback restores surface, `(ι, G)`, `fun`, `jacobian`, and
   stationarity reporting (`final_gradient_norm`,
   `inner_penalty_residual_l2`), and reports `success=False`. A
   non-finite Hessian does not veto a finite iterate. Rollback drops
   the last-iterate Hessian (`hessian is None`) instead of recomputing
   C++'s Hessian at the restored point; coil VJP still uses the
   operator path. Persist is on `‖∇J_LS‖₂`, not `J_LS`.

Post-Newton `J_LS`/`∇J_LS` are re-evaluated with the same cpu-ordered
closure as the packed-state check, not taken from JAX's internal Newton
`grad` (which iterates the default/jvp objective).

The CI surface is NCSX `SurfaceXYZTensorFourier` stellsym, `mpol=ntor=2`,
`7×7` quadrature. It is the same **operator** as F3 255×64, not the same
**scale**. Newton is locked after a native LBFGS seed (on-manifold polish,
reconstruct-start style) and after a `+1e-3` ι perturbation that forces
at least one Newton step. A cold 10-step `fit_to_curve` walk can still
differ from C++ because JAX Newton backtracks while C++ takes full
steps; that globalization is not this canary. Persist is on the public
adapter method `minimize_boozer_penalty_constraints_newton` (every
backend of that method). In-graph Newton inside
`simsopt_jax.geo.optimizers.optimizer` is unchanged.

## Explicitly not claimed

- F3 B37 nested-LS process-wall vs native banana.
- Inheritance of 7.70× / 1.67× / 7.36×.
- Reduced 661-state varpro Newton, IFT adjoint, or fused XLA inner.
- Three-point F3 (archived start, GPU B37, flat-native B37) JAX Newton.
  Reconstruct JSON still only has C++ Newton on start and GPU B37.
  Flat-native B37 is a second off-manifold point, not the nested timing bar.

## Next

Superseded as the active nested-LS plan by
`docs/receipts/nested_ls_reduced_track_20260820.md`. Canary-1 remains the
full-state operator lock. The reduced track owns QR `y*`, Newton on `s`,
and Gate 1 255×64 start harness. Do not launch B37 nested-LS timing until
that receipt's physics gates match. Do not inherit F3 7.70×.
