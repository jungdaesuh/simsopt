# Single-stage JAX GPU projected-HVP curvature canary

Status: **closed bounded-negative; no multi-step authorization**  
Route: `CFS-PHVP1`  
Date: 2026-08-10

## Purpose

Test the smallest second-order route that directly addresses the failed
`CFS-CURV1` mechanism. The closed canary proved that the exact scaled
Lagrangian Hessian can be differentiated accurately, but the unregularized
full-space KKT system was ill-conditioned and produced a destructive step.

This route keeps the same physics, objective, 716 variables, 255 exact
equalities, FP64 derivatives, optimizer scaling, dual convention, bootstrap,
and endpoint KKT audit. It changes only the subproblem solver:

- no dense Hessian materialization;
- no unregularized saddle-point inversion;
- exact Lagrangian Hessian-vector products inside a tangent-space trust region;
- negative and near-null curvature terminate safely at the trust boundary.

This is a one-step causal canary. It cannot establish convergence or a GPU
speed win.

## Frozen mathematics

Use the established scaled coordinates and plus-sign Lagrangian:

```text
z(u) = z0 + S u
c(u) = D q(z(u))
L(u, lambda) = Phi(z(u)) + lambda^T c(u)
r_d = grad Phi(u) + A^T lambda
A = dc/du
H v = jvp(grad_u L, u, v).
```

The scaled multiplier is the certified minimum-stationarity projection from

```text
(A A^T) lambda = -A grad Phi.
```

Let

```text
P v = v - A^T (A A^T)^-1 A v.
```

Both variants solve the same tangent trust-region model at the same bootstrap:

```text
minimize_t  r_d^T t + 0.5 t^T B t
subject to  A t = 0, ||t||_2 <= Delta.
```

`IDENTITY` uses `B v = v`. `EXACT_HVP` uses the exact scaled Lagrangian HVP.
Projected Steihaug CG stops on projected-residual convergence, boundary
crossing, or nonpositive curvature. The latter two cases move exactly to the
boundary; max-iteration exhaustion is unusable.

Freeze:

- `Delta = 2^-10`, selected before this run from the sealed `CFS-CURV1`
  evidence to avoid repeating its destructive `1/64` step;
- maximum 32 HVP/CG iterations;
- projected residual tolerance `1e-10`;
- tangency, linear-solve, and corrected-feasibility tolerances `1e-10`;
- estimated forward-error limit `1e-7`;
- one shared minimum-norm nonlinear correction using each trial-state Jacobian;
- full step `alpha = 1` for both variants, with no per-variant line search.

The length penalty is at its one-sided `max()` kink at the bootstrap. JAX's
frozen AD convention therefore defines a deterministic generalized curvature,
not a claim of a unique classical Hessian for that term.

## Why not Gauss--Newton in this canary

The objective can be residualized in principle, but no current canonical
residual-vector contract preserves the state-dependent non-QS normalization
and termwise FP64 reductions. More importantly, Gauss--Newton omits both
residual-weighted objective curvature and nonlinear equality curvature and is
positive semidefinite. It therefore cannot test the negative-curvature
mechanism implicated by `CFS-CURV1`. It remains a separate contingent route,
not the exact-curvature comparator here.

## Implementation boundaries

Add, without modifying closed route artifacts:

- a generic dense-constraint/callable-HVP projected Steihaug primitive;
- a fullspace adapter that reuses the established scaling, evaluator,
  multiplier projection, and independent physical endpoint diagnostics;
- a provenance-bound RTX 5090 runner and focused tests.

The timed executable must remain device resident. The inactive tail of the
fixed-shape CG loop must guard the expensive HVP with `lax.cond`, not merely
mask its result after execution.

## Fail-closed gate

The canary is usable only when all of the following hold:

- backend is exactly one RTX 5090 GPU, FP64 enabled, dimensions `716/255`;
- source manifest is identical before and after execution;
- initial, correction, and endpoint Gram solves are finite, with matrix-scaled
  residual `<=1e-10` and forward-error estimate `<1e-7`;
- exact HVP deterministic bilinear-symmetry defect `<=1e-10`;
- both tangent steps satisfy normalized `||A t||_inf <=1e-10` and
  `||t||_2 <= Delta` within FP64 rounding tolerance;
- both models report positive predicted reduction;
- interior termination meets the projected-residual tolerance; boundary or
  negative-curvature termination reaches the frozen boundary; exhaustion is
  unusable;
- corrected scaled feasibility is `<=1e-10`;
- endpoint raw physical KKT and objective are finite;
- zero hot H2D/D2H transfers and peak bound memory is below `0.8` of the GPU.

The exact-HVP hypothesis is supported only if both variants are usable and the
exact endpoint raw physical KKT stationarity is at most half both the bootstrap
and identity endpoint values. The objective must not increase relative to the
bootstrap. Otherwise the terminal status is
`NOT_SUPPORTED_BY_ONE_STEP_CANARY` or `CANARY_NOT_USABLE`.

Only `SUPPORTED_BY_ONE_STEP_CANARY` authorizes a separate bounded three-step
convergence canary. No radius, tolerance, iteration budget, or acceptance rule
may be tuned after inspecting this result.

## Done

- Generic projected-HVP tests cover SPD interior convergence, indefinite
  boundary exit, tangency, radius, certified correction, exact-HVP finite
  differences, and whole-program JIT compilation.
- The fullspace adapter test proves unchanged scaling and independent raw KKT
  endpoint recomputation.
- One fresh RTX 5090 artifact records runtime, source, compile/execution time,
  memory, transfers, HVP/CG telemetry, correction certificates, and the frozen
  terminal gate.
- The result is sealed read-only and this document records its hashes and
  disposition.

## Executed outcome — 2026-08-10

The provenance-bound RTX 5090 result is sealed at
`artifacts/cfs-phvp1-20260810T0901Z`. The result SHA-256 is
`f8bfe8ffe6609350b5d435e1e8c10fe8a1808a2b08a97ca87e49f12190492d09`;
the 1,940-entry source-manifest SHA-256 is
`120b8d87ed36d2be39b3d8cf6f8e7a8a1cd535f83c119227cdbc126bce3e6e0f`.
The directory is mode `0555` and both files are mode `0444`. Strict JSON,
hash, source-entry, transfer, memory, and terminal-gate checks pass.

The terminal status is **NOT_SUPPORTED_BY_ONE_STEP_CANARY**. Both variants
were usable, finite, feasible, and condition-certified, but exact HVP did not
meet the frozen twofold raw-KKT reduction:

| Quantity | Bootstrap | Identity endpoint | Exact-HVP endpoint |
|---|---:|---:|---:|
| Raw physical KKT stationarity | `3.0894326526207685e-02` | `3.0515656850430218e-02` | `3.034866720252498e-02` |
| Physical objective | `8.444212891013326e-05` | `8.399719186803576e-05` | `8.381261411456963e-05` |
| Scaled feasibility | `1.1145854108785588e-15` | `2.6750049861085414e-15` | `1.13687711909613e-14` |
| Model-step norm | `0` | `6.791192049552498e-04` | `9.765624999999999e-04` |

The exact-HVP step used one HVP, encountered no nonpositive curvature on its
first direction, and terminated at the frozen trust boundary. Its predicted
reduction was positive, its tangent residual was `2.351686239080974e-16`, and
the deterministic bilinear HVP symmetry defect was
`1.6605731779015443e-12`. The trial-state correction restored scaled
feasibility to `1.13687711909613e-14`.

Compilation took `129.218831448 s`; synchronized execution took
`2.750478076 s`. Peak process GPU memory was `24,822 MiB`, or
`0.7612475848744135` of the RTX 5090. The timed executable recorded zero hot
H2D and D2H transfers. These are diagnostic timings, not a speed-to-solution
claim.

Exact projected curvature improved raw KKT by only about 1.8% from the
bootstrap and about 0.55% relative to the identity endpoint. That is useful
directional evidence, but it is far below the predeclared causal gate. No
three-step run is authorized, and no radius or tolerance replay is permitted.
`CFS-PHVP1` is therefore closed as a bounded-negative one-step route.
