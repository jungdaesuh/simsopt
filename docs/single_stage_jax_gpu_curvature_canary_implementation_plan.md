# GPU-Native Single-Stage Curvature Canary

**Status:** `CLOSED / CANARY_NOT_USABLE / THREE_STEP_NOT_AUTHORIZED`  
**Route identity:** `CFS-CURV1`  
**Parent evidence:** closed `CFS-AL1/AL2`, `CFS-SQP1`, and `CFS-FTR1` routes

## Objective

Determine whether under-informed quasi-Newton curvature is the immediate cause
of poor projected-primal endpoint geometry in the coupled/fullspace routes.
This is a causal diagnostic, not another optimizer route, a test of the full
primal-dual state update, or a speed campaign.

The canary compares two directions at the identical feasible bootstrap:

1. `IDENTITY_BFGS`: the frozen bootstrap Hessian model `B = I_716`;
2. `EXACT_LAGRANGIAN`: the exact FP64 Hessian of the same scaled Lagrangian at
   the same state and consistently projected multiplier.

Everything else is shared: physics, 716 DOFs, 255 equalities, scaling,
derivatives, KKT right-hand side, trust-radius cap, nonlinear feasibility
correction, endpoint multiplier projection, and endpoint diagnostics.

## Mathematical contract

Use the existing optimizer coordinates and scaled equality convention:

```text
z(u) = z0 + S u
c(u) = D q(z(u))
L(u, lambda) = Phi(z(u)) + lambda^T c(u).
```

At the bootstrap, materialize the exact objective gradient `g` and equality
Jacobian `A`. Project the initial scaled multiplier using

```text
(A A^T) lambda = -A g.
```

For each curvature model `B`, solve exactly one linearized KKT direction:

```text
[ B  A^T ] [p]  = -[g + A^T lambda]
[ A   0  ] [d]     [c].
```

For `EXACT_LAGRANGIAN`, `B = d^2 L / du^2` is materialized with one linearized
gradient and exact-tail batched Hessian-vector products. The dense Hessian is
symmetrized only after its pre-symmetrization defect is recorded.

Cap both primal directions at the same optimizer-space radius `1/64`. Apply one
minimum-norm nonlinear feasibility correction using the shared bootstrap
Jacobian:

```text
u_trial = u0 + p_capped
(A A^T) w = c(u_trial)
u_corrected = u_trial - A^T w.
```

Recompute the full joint linearization at each corrected endpoint and project
its multiplier again before comparing raw physical KKT stationarity. Initial
and endpoint KKT values therefore use the same multiplier convention.

## Design-it-twice decision

1. Extending `CFS-FTR1` with a curvature callback would couple this diagnostic
   to the closed route's filter, radius updates, and BFGS state machine.
2. A standalone one-step A/B changes only the curvature model and reuses the
   same evaluator, scaling, correction, and endpoint diagnostics.

Select design 2. It isolates the hypothesis with fewer numerical decisions and
does not reopen or reinterpret frozen route evidence.

## Gate

The one-step canary is usable only when both variants satisfy all of:

- FP64 state with dimensions `716/255`;
- finite Hessian, directions, corrected states, and diagnostics;
- Hessian symmetry relative defect `<= 1e-10`;
- dense-Hessian action versus direct HVP relative defect `<= 1e-10`;
- KKT direction RHS-relative 1-norm residual `<= 1e-10` after one LU
  refinement;
- retained-LU Hager--Higham condition estimate with estimated forward-error
  bound `kappa_hat * eta / (1 - kappa_hat * eta) < 1e-7`;
- multiplier projection and nonlinear correction residuals `<= 1e-10`, with
  eigenvalue-based Gram forward-error bounds `< 1e-7`;
- corrected scaled feasibility `<= 1e-10`;
- zero hot-loop H2D and D2H transfers;
- peak bound GPU memory below `0.8` of the RTX 5090.

The curvature hypothesis is **SUPPORTED_BY_ONE_STEP_CANARY** only when the exact
variant is usable, the identity variant is usable, and exact curvature produces
raw physical KKT stationarity at most half both the consistently projected
bootstrap value and the identity endpoint. Otherwise the result is
**NOT_SUPPORTED_BY_ONE_STEP_CANARY** or **CANARY_NOT_USABLE**, with no tuning or
speed claim.

Only a supported one-step result authorizes a separate three-step curvature
canary. One or three steps cannot establish convergence or GPU speed-to-solution.

## Done criteria

- Generic dense-curvature primitives pass quadratic, nonlinear-equality,
  exact-tail, orientation, and residual tests.
- The fullspace adapter proves unchanged physics/scaling and independently
  recomputes raw endpoint diagnostics.
- One fresh, immutable, provenance-bound RTX 5090 result records source,
  runtime, bootstrap, memory, transfers, timing, direction/Hessian identities,
  and the terminal gate above.
- Prior `CFS-AL`, `CFS-SQP1`, and `CFS-FTR1` artifacts remain byte-identical.

## Executed outcome — 2026-08-10

The provenance-bound RTX 5090 run is sealed at
`artifacts/cfs-curv1-20260810T0249Z`. Its result and source-manifest SHA-256
values are, respectively,
`fc55179a7acc655ec07f985d97f3f6c82315d43d448c658dd1f9ac2fc9a2c8c3` and
`a34fa5585e3e05d27f89edb99a0a3576458a153ba02d940593fc796aaad11b0b`.
The result directory is mode `0555`; both files are mode `0444`. The manifest
contains 1,938 current-byte entries and was identical before and after the
run.

The terminal status is **CANARY_NOT_USABLE**:

| Quantity | Bootstrap | Identity endpoint | Exact endpoint |
|---|---:|---:|---:|
| Raw physical KKT stationarity | `3.0894326526207997e-02` | `3.0515656907548712e-02` | `5.349319415698302` |
| Scaled feasibility | `9.556199332406843e-16` | `1.2438884643945804e-11` | `3.428565000882693e-07` |
| Raw direction norm | `0` | `6.791192049552456e-04` | `4.635029216483018e+01` |
| KKT condition estimate | `1` | `2.852518761861507e+06` | `9.858477083808194e+10` |
| KKT relative residual | `0` | `8.770714329038105e-16` | `6.134608746553258e-10` |
| Estimated KKT forward-error bound | `0` | `2.5018627241101922e-09` | nonfinite |

The exact Hessian itself passed its construction checks: symmetry relative
defect `6.7391033675344875e-12`, action relative defect
`3.5266940439174083e-12`. The failure is the unregularized exact Newton--KKT
system and its step, not Hessian assembly. Its solve cannot meet the frozen
FP64 forward-accuracy gate, the single feasibility correction misses
`1e-10`, and raw KKT stationarity worsens by about 173 times. Therefore this
result does not support the claim that replacing identity/BFGS with an
unregularized dense exact Lagrangian Hessian fixes the convergence problem.

The synchronized execution took `5.563961264 s` after `94.328538038 s` of
compilation. Exact-process GPU memory peaked at `24,818 MiB`, or
`0.7611249118287484` of the physical RTX 5090, below the `0.8` gate. The timed
execution used `jax.transfer_guard("disallow")` and recorded zero hot H2D and
D2H calls. These timings are diagnostic only and are not a speed result.

No three-step canary is authorized. A future second-order route must handle
negative/near-null curvature inside a regularized projected trust-region or
matrix-free HVP subproblem; it must not replay this raw exact-KKT solve or
infer that another globalization wrapper around the same step will repair it.

## Native convergence correction

The `287.30421751597896 s` native record is a time-to-1,000-iteration
engineering threshold, not a converged endpoint. The preserved native receipt
(`8118529751f184f60f0c4d26f338cd1832aae579004d62866fb2a2f6617e9fe4`)
reports `budget_exhausted`, `outer_solver_success=false`,
`endpoint_certificate_success=false`, `terminal_stationary=false`, and
`constraints_satisfied=true`. Its final objective
`4.4822246533126125e-08` remains a useful quality target, but it is not evidence
that the C++/native route satisfied the present fullspace KKT gate.
