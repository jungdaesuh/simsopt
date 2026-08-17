# Finite-build native-lane fp32 taint — diagnostic record (2026-08-17)

Archived evidence for the amendment of the same date in
`jax_gpu_finitebuild_native_speed_implementation_plan.md`. All numbers below
are verbatim probe outputs from the session that root-caused the defect; the
two published rows they interrogate are permanent artifacts:

- `.artifacts/stage_two_finitebuild_native_gpu/20260817T183358Z-jax-sweep-2352781/rows/jax-sweep-h20-b400.json`
  (sha256 `80bc1bb4c297fee7…`) — the GPU leg's self-report at its endpoint.
- `.artifacts/stage_two_finitebuild_native_gpu/20260817T183358Z-jax-sweep-2352781/rows/native-endpoint-jax-sweep-h20-b400.json`
  (sha256 `3bb75441d6fd4fba…`) — the native oracle re-evaluation of the same
  solution vector, produced under the then-unpinned environment.

## Symptom

The jax-sweep phase reduced to `NOT_PRODUCED`:
`gradient_inf_norm (3.168897559317626e-05 vs 2.909781209681831e-05)` — an
8.9% lane disagreement against the `rtol=0.05` cross-check. Component-wise
(147 DOFs): median `|Δ| = 2.68e-09`; 18 components exceeded `5e-7`,
localized to DOF indices 38–68, maximum `|Δ| = 2.5912e-06` at index 46 (the
inf-norm argmax of both lanes). Objective disagreement at the same x:
`3.07e-13` absolute = `3.0e-07` relative on `J ≈ 1.012e-06` — itself the
fp32-epsilon signature. The `2.6e-6` figure cited by the plan amendment is
this maximum per-component lane gap.

## Probe 1 — native self-spread (kills the summation-noise hypothesis)

Five re-runs of the harness's own `native-endpoint-eval` leg at the same x,
separate processes, OMP ∈ {1, 1, 2, 8, 8}, then the original oracle row:

```
inf norms: all six native draws = 2.909781210e-05   (bitwise: max spread 0.000e+00)
jax-self                        = 3.168897559e-05
```

The native evaluator reproduces itself bit-for-bit across thread counts and
processes; the disagreement is not floating-point summation order.

## Probe 2 — two-lane FD arbitration at DOF 46 (convicts one analytic gradient)

Central differences of each lane's own objective along `e_46`,
`h ∈ {2, 3, 4, 6, 8}e-6`, in an environment with `JAX_ENABLE_X64=true`:

```
value identity at x: f_nat=1.012357557685887e-06  f_jax=1.012357557685882e-06
analytic native g[46]      = 3.168897558e-05
jax adjoint GPU row g[46]  = 3.168897559e-05
jax adjoint CPU here g[46] = 3.168897559e-05
h=2e-06: fd_nat 4.311724261e-05  fd_jax 4.311724261e-05  |f diff| ~1e-20/point
h=3e-06: fd_nat 5.741343203e-05  fd_jax 5.741343203e-05
h=4e-06: fd_nat 7.744836661e-05  fd_jax 7.744836661e-05
h=6e-06: fd_nat 1.348214114e-04  fd_jax 1.348214114e-04
h=8e-06: fd_nat 2.154684625e-04  fd_jax 2.154684625e-04
Richardson fit fd(h)=g+c·h²: g = 3.153986562e-05, c = 2.873e+06,
                             max fit residual 1.33e-07  (both lanes identical)
```

The two objectives agree to ~1e-20 at every probed point, so they share one
FD limit; the fit (residual `1.3e-7`) is consistent with `3.169e-05` and
excludes `2.910e-05` (forcing `g = 2.910e-05` misfits `h=8e-6` by `~4e-5`).
Note the clean-environment native `dJ` itself returns `3.168897558e-05`,
matching the JAX adjoint to `1e-14` relative — the oracle rows' value is not
reproducible once x64 is enabled.

## Probe 3 — call-order minimal pair (kills the cache hypothesis)

Fresh evaluator per scenario, x64 environment: every pre-`dJ()` call pattern
(none; `J()`; `flux.J()`; `length_term.J()`; `distance_term.J()`;
`shortest_distance()`; per-coil `CurveLength.J()`; the exact oracle order)
returns `g[46] = 3.168897558e-05`. Call order is not the discriminator. A
sorted-multiset comparison of the two gradients also refutes a DOF-ordering
permutation (max sorted difference equals the raw gap).

## Probe 4 — the environment discriminator (the root cause)

The only material difference between the disagreeing probe processes was
`JAX_ENABLE_X64`. One-variable rerun of the exact oracle diagnostic path:

```
WITH x64: unscaled_state g[46] = 3.168897558e-05    (clean)
WITHOUT : (all oracle legs)      2.909781210e-05    (corrupted)
```

`_leg_environment` scrubs inherited `JAX_*` and re-pinned only
`JAX_PLATFORMS=cpu` for native legs; `simsopt.geo` objectives are jax-jitted,
so the native lane's transitively imported JAX pieces ran float32.

## Baseline fail-open measurement

`validate_run` on both pre-fix baseline directories returns `IDENTITY_OK`
under the defective environment. Measured native-vs-JAX gradient agreement
there: max `|Δ| = 2.494e-09` on components of `3.5e-02` (`~7e-08` relative —
fp32-epsilon, about seven orders above fp64-lane agreement), admitted by
`BASELINE_GRADIENT_RTOL = 1e-6` / `BASELINE_GRADIENT_ATOL = 1e-8`. The
baseline identity gate is therefore not a precision detector; the structural
detector is the observed-`jax_enable_x64` conformance clause added with the
fix.
