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

Post-Newton `J_LS`/`∇J_LS` are re-evaluated with the same cpu-ordered
closure as the packed-state check, not taken from JAX's internal Newton
`grad` (which iterates the default/jvp objective).

The CI surface is NCSX `SurfaceXYZTensorFourier` stellsym, `mpol=ntor=2`,
`7×7` quadrature. It is the same **operator** as F3 255×64, not the same
**scale**. Newton is locked after a native LBFGS seed (on-manifold polish,
reconstruct-start style) and after a `+1e-3` ι perturbation that forces
at least one Newton step. A cold 10-step walk from `fit_to_curve` is
**not** locked: native rolls back on a worse gradient, JAX does not, so
the endpoints diverge. That cold walk is the F3 B37 case and remains open.

## Explicitly not claimed

- F3 B37 nested-LS process-wall vs native banana.
- Inheritance of 7.70× / 1.67× / 7.36×.
- Reduced 661-state varpro Newton, IFT adjoint, or fused XLA inner.
- Off-manifold 10-step Newton walks (native rollback vs JAX continue).
- Three-point F3 (archived start, GPU B37, flat-native B37) JAX Newton.
  Reconstruct JSON still only has C++ Newton on start and GPU B37.
  Flat-native B37 is a second off-manifold point, not the nested timing bar.

## Next

Three-point F3 fixed-point on 255×64 using this pair API, then derivative
identities, then predictor–corrector cost. Do not launch B37 nested-LS
timing until those pass.
