# GPU-Native SQP/Primal-Dual Single-Stage Implementation Plan

**Status:** Approved for execution — Revision 2
**Last updated:** 2026-08-09
**Execution role:** Single source of truth for the `CFS-SQP1` tranche

**Revision 2 decision (2026-08-09):** Revision 1's static reciprocal-condition
floor is removed from the live KKT-validity policy. The immutable Revision 1
campaign remains historical evidence and is not reinterpreted. Revision 2 gates
the computed forward-error certificate directly and requires a fresh derivative
artifact before any downstream gate.

## Purpose

Implement and evaluate one GPU-native equality-constrained SQP route for the
single-stage problem. The route must directly reduce the original feasibility
and KKT residuals while exploiting batched first derivatives, dense GPU linear
algebra, and device-resident control. It may use a different optimizer and
trajectory from C++/native, but it may not change the physics or mathematical
problem.

This tranche follows the completed `CFS-P0`/`CFS-AL1`/`CFS-AL2` investigation.
Those routes proved that a device-resident JAX solve can execute quickly, but
the augmented-Lagrangian/L-BFGS family did not converge. Their source records,
artifacts, and `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING` conclusion are frozen
inputs, not routes to replay or reinterpret.

## Terminal objective

The tranche has exactly two valid terminal dispositions:

1. `ENGINEERING_SPEED_GOAL_ACHIEVED`: a provenance-valid, independently
   certified `CFS-SQP1` endpoint whose three fresh warm synchronized GPU solves
   each take less than `287.30421751597896 s`.
2. `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`: raw evidence identifies the exact
   correctness, convergence, memory, compilation, transfer, or projected-time
   gate that rejected `CFS-SQP1`.

The formal comparative verdict remains `NOT_PRODUCED` unless a separate
matched-baseline or baseline-import contract is independently satisfied.

## Goals

- Preserve the exact FP64, 716-variable, 255-equality fullspace problem.
- Replace penalty-driven convergence with a direct SQP KKT step.
- Materialize objective and equality first derivatives from one primal graph
  using exact-tail batched reverse rows on GPU.
- Keep optimizer, BFGS, multiplier, factorization selection, globalization,
  termination, and counters on device in one compiled dispatch.
- Produce route-specific, tamper-evident result, memory, endpoint-certificate,
  sample-receipt, and campaign artifacts.
- Run only the minimum derivative, one-step, ten-step, cold, and conditionally
  warm evidence required by the gates below.

## Non-goals

- Replaying, retuning, overwriting, or promoting `CFS-P0`, `CFS-AL1`,
  `CFS-AL1-B`, or `CFS-AL2`.
- Rerunning the native CPU optimizer or creating a new native baseline.
- Matching the C++ optimizer trajectory or line-search decisions.
- Weakening objective, feasibility, KKT, branch, cross-evaluator, field-line,
  fixed-state, FP64, or provenance requirements.
- Exact dense Lagrangian-Hessian materialization in the base route.
- First-order Uzawa/primal-dual gradient iteration.
- Automatic matrix-free Newton-Krylov fallback. That requires a later explicit
  SSOT revision and an exact-HVP/preconditioner gate.
- Mixed-precision authoritative state, derivatives, factors, or results.
- Production/default-route promotion during this tranche.

## Frozen prior evidence

The following authoritative bytes must remain unchanged:

| Artifact | SHA-256 | Disposition |
| --- | --- | --- |
| `docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md` | `17350f221664b05be5b97d36da519162a36c4bff1649b54f32027523f40f06d1` | Prior execution SSOT |
| `docs/single_stage_jax_gpu_coupled_fullspace_results.md` | `47eab59c8f19b33a74063aebe29e31614f0fe15fc6d8d63bffeed506f52fc939` | Prior bounded-negative result |
| `CFS-P0` RTX 10-step result | `0a97b54d31cab9cea015f804ee7f7efd05274c071aa135896177dd19218ed783` | Prior diagnostic result |
| `CFS-P0` RTX 100-step result | `b66eed04c372cc2d8c02649c9cc7d60b449c96797e1c63f49325c1f0e3d2fb85` | Prior diagnostic result |
| `CFS-AL1` read-only result | `617bc9ca71e17da2d64ff67358a1953ce96b885ea4e5a7f67687ccf5556cf1f0` | `SOLVER_RESULT_NOT_CONVERGED` |
| `CFS-AL2` read-only result | `1eb3998aea4e3f64e0a0e328b03afca6fee022d32bebaca78ca1f9e3051bff53` | `SOLVER_RESULT_NOT_CONVERGED` |

Before any route-enum expansion, Phase 0 writes the canonical immutable
`docs/single_stage_jax_gpu_sqp_prior_campaign_manifest.json`. It records the
path, file-byte SHA-256, size, and mode of the prior plan and results SSOTs and
of every bootstrap artifact, runtime-evidence document, source manifest, and
raw result in the authoritative RTX P0-10, P0-100, AL1-cold, and AL2-cold
campaign roots. Those roots contain route-specific results rather than a
complete `campaign.json`; absence of a prior campaign receipt is recorded, not
synthesized. The manifest itself is verified at Phase 0 and again at terminal
closure.

The strict-parity example and existing production/default route remain
unchanged. Existing unrelated dirty and untracked work is user-owned and must
not be reverted, reformatted, committed, or included in claim-bearing source
snapshots unless it is an actual runtime dependency.

## Authoritative mathematical problem

Reuse the established fullspace ledger without modification:

- physical state `z` has 716 free coordinates: 461 coil coordinates, 253
  surface coordinates, `iota`, and `G`;
- the first base current and every existing fixed coil/surface component remain
  outside `z`;
- `q(z)` contains the 254 masked Boozer residual components followed by the
  signed volume-label residual;
- `Phi(z)` is exactly the weighted sum of the non-QS surface ratio, the
  normalized full Boozer-residual scalar, `0.5*(iota-iota_target)^2`,
  `0.5*(major_radius-major_radius_target)^2`, and
  `0.5*max(total_length-length_target,0)^2`, with the same targets, weights,
  signs, grids, quadrature, and physical constants;
- curvature, curve-curve, curve-surface, and surface-vessel terms retain their
  frozen zero weights; and
- the complete endpoint equations are

```text
q(z) = 0
grad_z Phi(z) + J_q(z)^T lambda_raw = 0.
```

Use the existing centered variable and equality scaling:

```text
z(u) = z0 + S u
c(u) = D q(z(u))
lambda_raw = D^T lambda_scaled
A(u) = dc/du = D J_q(z(u)) S.
```

`z0` is the authoritative exact bootstrap. A complete run starts at `u=0`.
The changed-state canary uses the already frozen direction
`u_changed = 1e-3 * linspace(-1,1,716) / ||linspace(-1,1,716)||_2`.
The canonical JSON envelope containing the nine rows of `TERM_LEDGER` has
SHA-256 `2f0da302ae54441250bbb7f89f56c0e1540eed2e3ad16ea7af60d3bad505e5a0`;
Phase 0 and terminal closure must reproduce this digest.

## Design classification and design-it-twice decision

This is a Tier-4 integrity-sensitive additive schema and Tier-3 public solver
change. The user's instruction to create and execute this SSOT is the required
sign-off. Rollback is removal of the opt-in SQP route and campaign-v2 readers;
the legacy routes and v1 artifacts remain readable and byte-identical.

Two designs were evaluated:

### Design A — exact first derivatives, dense Schur SQP, damped BFGS

At iteration `k`, compute objective gradient `g`, scaled constraints `c`, and
their exact Jacobian `A`. With scaled equality multipliers `lambda` and an SPD
Powell-damped BFGS matrix `B`, form

```text
r_d = g + A^T lambda
r_p = c.
```

Solve the exact linearized KKT equations through the SPD Schur complement:

```text
B W = A^T
B v = r_d
(A W) delta_lambda = c - A v
p = -B^-1 (r_d + A^T delta_lambda).
```

This is algebraically equivalent to solving

```text
[ B  A^T ] [ p            ] = -[ r_d ]
[ A   0  ] [ delta_lambda ]   [ c   ].
```

It requires dense matrices of only about 7 MiB in total authority data, uses
GPU Cholesky/multiple-RHS solves, and avoids repeated dependent Krylov products.

### Design B — matrix-free exact-Hessian primal-dual Newton-Krylov

This applies the operator `(H_L v + A^T w, A v)` using exact HVP/JVP/VJP
products inside MINRES/GMRES. It avoids dense storage, but 64–128 Krylov steps
would require roughly 192–384 dependent derivative products per SQP step. The
repository has no certified fullspace exact-Hessian path or block
preconditioner, and prior evidence identifies dependent operator traversals as
a launch/conditioning risk.

### Decision

Implement Design A as the sole authorized route `CFS-SQP1`. Design B, exact
dense Hessians, and first-order primal-dual iteration remain unselected. This
keeps the implementation first-order in physics derivatives while using SQP to
target the original KKT equations directly.

## Frozen `CFS-SQP1` algorithm

### Joint derivative materialization

Define one pure function returning `(Phi(u), c(u))`. Use one `jax.vjp` primal
evaluation and apply its pullback to the 256 output basis rows in exact-tail
batches of 8. The resulting `256 x 716` matrix has `g` in its first row and
`A` in its remaining 255 rows. Physical rows are never padded. The batch width
is part of the route identity and may not be retuned from a complete endpoint.

The derivative primitive must prove:

- one primal evaluation per accepted SQP linearization;
- equality with independent `jax.grad`/`jax.jacrev` on small problems;
- objective directional finite differences;
- `A v`/`A^T w` transpose identity at bootstrap and changed state;
- exact `(255,716)` ordering and FP64 dtype; and
- JIT nesting with no host callback, scalar extraction, or D2H transfer.

### Dense KKT step

- Initialize `B0 = I_716` and `lambda0 = 0` in scaled coordinates.
- Maintain `B` with Powell-damped BFGS using curvature fraction `0.2`.
- Form the BFGS secant difference from Lagrangian gradients at the old and new
  primal states using the same accepted new multiplier.
- Define `B_delta = B + delta I` and reuse one Cholesky factor of `B_delta` for
  `A^T`, `r_d`, and the final primal step.
- Try the fixed regularization ladder
  `(0, 1e-12, 1e-10, 1e-8, 1e-6)` and select the first finite candidate whose
  normalized reconstructed KKT residual is at most `1e-10`.
- Require the Schur matrix `S_delta = A B_delta^-1 A^T` to have a finite
  Cholesky factor and a normalized solve residual at most `1e-10`.
- Record `min(abs(diag(L))) / max(abs(diag(L)))` for both the `B_delta` and
  `S_delta` Cholesky factors as diagnostic telemetry only. A Cholesky diagonal
  ratio is not a rank or conditioning certificate.
- Construct the exact symmetric `K_delta` corresponding to the solved system
  and compute its FP64 eigenvalues on device. Define
  `rho_K = min(abs(eigvalsh(K_delta))) / max(abs(eigvalsh(K_delta)))` as
  diagnostic evidence; do not impose a standalone reciprocal-condition floor.
- For the returned `x_hat = (p, delta_lambda)`, require the solution-scaled
  two-norm residual
  `zeta_2 = ||K_delta x_hat - b||_2 /
  (||K_delta||_2 ||x_hat||_2) <= 1e-10`. A zero denominator yields `0` only
  when the residual norm is also zero; this is accepted separately as the exact
  zero solution. Otherwise a zero denominator yields infinity.
- Require strict `rho_K > zeta_2` and directly gate the certified bound
  `zeta_2 / (rho_K - zeta_2) < 1e-7`. For
  `||K_delta||_2 ||x_hat||_2 > 0`, this proves
  `||x_hat - x||_2 / ||x||_2 <= zeta_2 / (rho_K - zeta_2)`.
  The independent raw physical KKT endpoint certificate remains mandatory.
- Never drop an equality, silently use a pseudoinverse, or reuse factors after
  the state or BFGS matrix changes.
- If every regularization candidate fails, terminate
  `RANK_DEFICIENT_OR_UNSTABLE_KKT`.

For candidate `x = (p, delta_lambda)`, define

```text
K_delta = [[B_delta, A^T], [A, 0]]
b = -[r_d, c]
kkt_relative_residual = ||K_delta x - b||_inf /
    max(1, ||K_delta||_inf ||x||_inf + ||b||_inf).
```

The infinity-norm residual remains compatibility telemetry. It does not replace
the unclamped solution-scaled two-norm certificate above.

The Schur right-hand side is
`h_delta = c - A B_delta^-1 r_d`, and its relative residual is

```text
||S_delta delta_lambda - h_delta||_inf /
    max(1, ||S_delta||_inf ||delta_lambda||_inf + ||h_delta||_inf).
```

Both norms are formed from the actual regularized matrices used by the solve.

The diagnostic derivative gate also computes singular values of `A` at the
bootstrap and changed state. It requires numerical row rank 255 with
`sigma_min > 1e-12 * sigma_max`. SVD is diagnostic only and is not in the hot
solve.

### Globalization and state update

Use the exact scaled-equality l1 merit

```text
merit_mu(u) = Phi(u) + mu ||c(u)||_1.
```

Freeze:

- `mu0 = 1`;
- `mu_next = max(mu, ||lambda + delta_lambda||_inf + 1)`;
- Armijo coefficient `eta = 1e-4`;
- step candidates `(1, 1/2, ..., 1/1024)`;
- candidates evaluated sequentially inside a fixed-shape device `lax.scan` to
  bound memory; and
- the current and every trial merit are both evaluated with the same
  `mu_next`; and
- with `d_mu = g^T p + mu_next*(||c + A p||_1 - ||c||_1)`, acceptance of the
  first finite candidate satisfying
  `merit_mu_next(u + alpha*p) <= merit_mu_next(u) + eta*alpha*d_mu`.

Update `u` and `lambda` with the same accepted `alpha`. Reject nonfinite trials
on device and count them. A nonfinite or nonnegative `d_mu` is a globalization
failure. If no candidate is accepted, retry once with `B=I` under the same KKT
and merit rules. A second failure terminates
`GLOBALIZATION_FAILED`; it does not silently accept a nondecreasing step.

### BFGS and termination

For accepted `s = u_next-u`, use

```text
y = grad_u L(u_next, lambda_next) - grad_u L(u, lambda_next).
```

Let `Bs = B s`, `a = s^T B s`, and `b = s^T y`. Freeze

```text
theta = 1                   if b >= 0.2*a
theta = 0.8*a/(a-b)         otherwise
y_bar = theta*y + (1-theta)*Bs
B_next = B - (Bs Bs^T)/a + (y_bar y_bar^T)/(s^T y_bar)
B_next = 0.5*(B_next + B_next^T).
```

The update uses the unregularized `B`, not `B_delta`. Every denominator and
the complete updated matrix must be finite, with `a > 0` and
`s^T y_bar > 0`, before division. Failure resets `B_next=I` and increments the
consecutive-reset counter. A successful non-reset update clears that counter;
two accepted iterations in a row requiring reset terminate
`BFGS_UPDATE_FAILED`.

Freeze a maximum of 100 SQP iterations and 1200 joint value/constraint
evaluations. A normal success requires all of:

- physical objective `<= 4.4822247e-08`;
- scaled feasibility infinity norm `<= 1e-10`;
- independently formed raw physical KKT stationarity infinity norm `<= 1e-7`;
- finite FP64 primal, dual, derivative, factor, and observable state; and
- no fatal solver status.

Reaching a KKT point above the objective-quality threshold is
`OBJECTIVE_QUALITY_REJECTED`, not success. Reaching either budget is
`ITERATION_LIMIT` or `EVALUATION_LIMIT`, not success.

### Device program and telemetry

The 100-iteration solver is one `jax.jit`-compiled fixed-shape `lax.while_loop`
or masked `lax.scan`. Its carry owns primal coordinates, scaled multipliers,
`B`, merit penalty, counters, finite latches, reset count, and last accepted
diagnostics. The hot loop has no Python callback, progress observer, host scalar,
mutable closure substitution, synchronization, H2D, or D2H.

The typed SQP result is distinct from `CfsAl1Result` and records:

- optimizer and physical coordinates and scaled/raw multipliers;
- objective, scaled/raw feasibility, raw KKT stationarity;
- normalized termination code and fatal/converged flags;
- iteration, joint-evaluation, derivative-build, KKT-solve, line-search,
  rejected-nonfinite, BFGS-reset, and regularization counters;
- final KKT solve residuals, reciprocal condition, Cholesky pivot diagnostics,
  and selected regularization; and
- fixed-shape accepted-iteration history for objective, feasibility,
  stationarity, step length, KKT residual, and status.

## Resources and staged execution gates

Use the RTX 5090 as the sole execution device for this tranche. No A100 or
native optimizer run is required.

| Boundary | Frozen limit |
| --- | ---: |
| Exact-row VJP batch width | 8 |
| Derivative/one-step containing process | 300 s |
| Ten-step containing process | 600 s |
| Complete containing process | 900 s |
| Warm synchronized solve | 287.30421751597896 s |
| Complete SQP iterations | 100 |
| Joint evaluations | 1200 |
| Whole-child peak GPU memory | `< 0.8 * physical device memory` |
| Hot transfers | 0 H2D, 0 D2H |
| Solve boundary transfers | one initial H2D, one final D2H |

The one-step canary starts only at `u_changed`. The ten-step scaling canary
starts only at the canonical bootstrap `u=0`; neither continues the other's
state. The ten-step dispatch is compiled first, compilation is reported
separately, and its synchronized execution is timed once. Freeze

```text
projected_100_iteration_s = 10 * measured_10_iteration_s.
```

Endpoint-audit and compilation time are reported separately and are not part
of the synchronized-solve threshold. This projection is a route-selection
gate, never a measured complete-solve result.

The ten-step scaling gate proceeds to a complete cold run only when:

- all correctness, rank, KKT residual, finite-state, and transfer checks pass;
- objective decreases from the canonical bootstrap;
- feasibility remains within the frozen tolerance or decreases from its
  independently measured bootstrap value;
- raw KKT stationarity decreases;
- measured memory remains below the whole-child limit; and
- `projected_100_iteration_s` remains below `287.30421751597896 s`.

The projection record must state its formula and raw inputs. It is a route gate,
not a speed result.

## Artifact and compatibility contract

All SQP artifacts are additive and live in new campaign roots. Never write into
the authoritative AL campaign roots.

Required schemas:

- `single-stage-fullspace-cfs-sqp1-result-v1` for raw non-certifying solver
  output;
- `single-stage-fullspace-cfs-sqp1-gpu-memory-v1` for parent-observed exact-PID,
  exact-device-UUID whole-child samples;
- `single-stage-fullspace-cfs-sqp1-endpoint-certificate-v1` for independent
  endpoint evidence;
- `single-stage-fullspace-cfs-sqp1-sample-receipt-v1` binding raw result,
  runtime, bootstrap, memory, and certificate artifacts; and
- `single-stage-fullspace-campaign-v2` binding the cold and, only after cold
  certification, exact `warm-1`/`warm-2`/`warm-3` sequence.

Adding `CFS-SQP1` to a route enum must not change legacy v1 campaign semantics
or hashes. Freeze an explicit legacy-v1 route order and contract digest, add
schema-dispatched v2 readers/writers, and add golden compatibility tests. The
runner must use exhaustive route dispatch; no new route may fall through to an
AL publisher.

The pre-expansion legacy authorities are frozen as:

- `LEGACY_V1_ROUTES = (CFS-P0, CFS-AL1, CFS-AL2, CFS-AL1-B)`;
- canonical `single-stage-fullspace-routes-v1` bytes: 5599 bytes, SHA-256
  `1cac4bd571dac722ae188693b26ab6cc86d2c5ca64f274f2a5b962a625a7b01b`;
- canonical `single-stage-fullspace-campaign-v1` contract bytes: 10722 bytes,
  SHA-256
  `e680e6a2f6ff0afb9bdcc18e15bf90953b77e7c92baaed1745a4d1008700e4f9`.

`CFS-SQP1` uses a distinct typed SQP policy rather than populating inapplicable
AL/L-BFGS fields. V1 readers, writers, validators, route ordering, and digest
functions remain explicitly pinned to the four-route legacy tuple. V2 has its
own explicit route order, payload, digest, parser, writer, and validator.

Each isolated sample records immutable source manifest, runtime/import/device
identity, bootstrap identity, compilation/cache state, synchronized solve time,
total child wall time, transfer audit, exact solver result, and parent-observed
memory. Raw solver output has `endpoint_certificate=null` and
`promotion_eligible=false`; only the independent certificate/receipt layer may
promote it.

## Endpoint certification

Reuse the existing scientific recomputation, but add an SQP-specific
termination adapter rather than fabricating AL stages. The certificate must
independently recompute and bind:

- objective and every raw objective term;
- 254 Boozer residuals plus the signed volume residual in frozen order;
- scaled feasibility and raw multiplier orientation;
- `grad Phi + J_q^T lambda_raw` and raw KKT infinity norm;
- fixed current and fixed geometry DOFs;
- inactive hardware metrics and zero weights;
- pre/post-projection state, with both independently certifiable and any
  projection change immaterial;
- native evaluation of the JAX endpoint and JAX evaluation of the historical
  native endpoint, without rerunning the native optimizer;
- authoritative exact-solve branch reproduction seeded at the candidate;
- optimizer-independent Poincare closure and traced-iota evidence; and
- basin classification when endpoints materially differ.

The cross-evaluator, branch, and field-line producers must record full
configuration, endpoint hashes, tolerances, raw values, runtime/source identity,
and artifact digests. Summary Booleans alone are insufficient.

## Implementation map

| Knowledge owner | Planned path and responsibility |
| --- | --- |
| Generic SQP mechanics | `src/simsopt_jax/geo/optimizers/dense_sqp.py`: typed prepared callback-free solver, joint VJP rows, Schur KKT solve, globalization, BFGS, telemetry |
| Fullspace SQP adapter | `src/simsopt_jax/solve/fullspace_sqp.py`: frozen problem/scaling adapter, SQP-specific result, raw-dual conversion, final KKT diagnostics |
| Route/legacy contract | `src/simsopt_jax/solve/fullspace.py`: additive route identity and v2 payload while preserving explicit v1 bytes |
| Scientific certificate | `src/simsopt_jax/solve/fullspace_certificate.py`: route-discriminated AL/SQP termination feeding one scientific endpoint checker |
| GPU runner | `benchmarks/run_single_stage_fullspace_gpu.py`: SQP-only probe, immutable child, synchronized timing, transfer evidence, exhaustive dispatch |
| Memory and receipts | `benchmarks/process_gpu_monitor.py`, `benchmarks/single_stage_fullspace_receipt.py`: exact PID/UUID memory and additive sample/campaign-v2 contracts |
| Endpoint audits | `benchmarks/single_stage_fullspace_endpoint_audit.py`: cross-evaluator, branch, field-line, projection, and certificate producer |
| Validation | `benchmarks/validate_single_stage_fullspace_campaign.py`: schema-dispatched v1/v2 validation and complete sample-chain enforcement |
| Results SSOT | `docs/single_stage_jax_gpu_sqp_primal_dual_results.md`: dated evidence ledger and terminal disposition |

## Implementation phases

### Phase 0 — Contract and frozen baseline

- [ ] Record live HEAD, dirty-tree identity, intended file set, environment,
  device, prior plan/result hashes, and AL artifact hashes.
- [ ] Add the machine-readable SQP budget/algorithm payload and canonical digest.
- [ ] Freeze explicit legacy-v1 routes and golden v1 contract bytes before enum
  expansion.
- [ ] Write and seal the prior-campaign manifest covering P0-10, P0-100, AL1,
  AL2, their bootstrap/runtime/source evidence, and prior SSOT/result bytes.
- [ ] Add `CFS-SQP1` and campaign-v2 request/result/status contracts.
- [ ] Add rollback, API-evolution, caller-inventory, migration, and observable-
  behavior sections to the results record.

Gate: prior hashes and v1 golden fixtures pass; no production/default caller
changes; all SQP knobs above are machine-readable before implementation.

### Phase 1 — Generic dense SQP core

- [ ] Implement the typed joint-VJP row materializer with exact tails.
- [ ] Implement Cholesky reuse, Schur solve, regularization selection, and the
  direct full-KKT forward-error/residual certificate.
- [ ] Implement Powell-damped BFGS, identity reset, l1 merit, and device Armijo.
- [ ] Implement one compiled prepared SQP program and typed fixed-shape result.
- [ ] Test observable behavior on independent equality-constrained quadratic and
  nonlinear problems, including singular, nonfinite, line-search, reset, and
  budget failures.

Gate: synthetic problems reach independently calculated primal/dual solutions;
JIT/transfer checks pass; singular systems fail closed; no private AL or solver
API is imported.

### Phase 2 — Fullspace derivatives and adapter

- [ ] Implement the pure `(Phi,c)` optimizer-coordinate evaluator.
- [ ] Prove the 256-row materialization against independent derivatives.
- [ ] Prove scaled/raw dual and stationarity identities.
- [ ] Measure bootstrap and changed-state rank/condition and one reconstructed
  SQP KKT step on CPU reference and RTX.
- [ ] Implement SQP-specific endpoint result and certificate normalization.

Gate: FP64 derivative, transpose, rank, KKT-residual, state-order, fixed-DOF,
and same-state physics tests pass with no nested Boozer solve or implicit
adjoint in the hot graph.

### Phase 3 — Runner, receipts, and endpoint producers

- [ ] Add SQP raw result, memory, certificate, sample-receipt, and campaign-v2
  schemas without changing legacy v1 validation.
- [ ] Add exhaustive route dispatch and immutable snapshot-child execution.
- [ ] Add exact PID/device memory monitoring outside the child graph.
- [ ] Implement cross-evaluator, branch, field-line, projection, and certificate
  producers with full provenance.
- [ ] Test cold-failure warm prohibition, exact sample order, no replacement,
  mixed-source/device/bootstrap rejection, tamper rejection, and relocation.

Gate: writer-produced synthetic success validates; every missing/tampered or
partial artifact fails closed; AL golden bytes and authoritative hashes remain
unchanged.

### Phase 4 — Minimal RTX gates

- [ ] Run one immutable changed-state derivative/rank/KKT-step artifact.
- [ ] Run one immutable one-step finite/acceptance/transfer artifact.
- [ ] Run one immutable ten-step scaling/memory/timing artifact.
- [ ] Apply the measured-path kill rule once; do not repeat a failed gate.

Gate: the ten-step criteria under Resources pass and the complete path remains
credible below the native engineering threshold.

### Phase 5 — Cold endpoint and conditional warm campaign

- [ ] Run exactly one complete cold sample from the canonical bootstrap.
- [ ] If its solver result is not converged, close bounded-negative and do not
  run endpoint audits or warm samples.
- [ ] If converged, run the complete independent endpoint audit and seal the
  cold certificate.
- [ ] If the cold certificate fails, close bounded-negative and do not run warm
  samples.
- [ ] Only after cold certification, run exactly `warm-1`, `warm-2`, and
  `warm-3` in fresh isolated processes and independently certify each.
- [ ] Produce and independently validate the campaign-v2 receipt.

Gate: every warm endpoint certifies and every synchronized warm solve is below
`287.30421751597896 s` for an engineering success. Any missing or failed gate
is non-promoting.

Each warm child has one frozen sequence: load pristine canonical inputs,
prepare and compile, execute one untimed solve from those pristine device
inputs and synchronize it, discard its result, restore identical pristine
device inputs without reusing optimizer carry, then time and synchronize
exactly one solve. Only the timed solve's endpoint is certified. A child that
times out or fails at any point is not replaceable.

### Phase 6 — Terminal disposition

- [ ] Record every attempted gate, raw artifact, exact hash, source/runtime
  identity, and failure or success mechanism.
- [ ] Recheck prior SSOT and AL artifact hashes byte-for-byte.
- [ ] Prove production/default routing remained unchanged.
- [ ] Run a fresh numerical/certificate review and a fresh provenance/receipt
  review.
- [ ] Record exactly one terminal disposition and keep the formal verdict
  `NOT_PRODUCED` unless separately authorized evidence changes it.

## Validation plan

Run JAX pytest files in separate CPU processes unless a test is explicitly a
GPU gate. Minimum suites:

- [ ] generic dense-SQP solver and result contract;
- [ ] fullspace joint derivative, scaling, and KKT identities;
- [ ] SQP route, certificate, and legacy compatibility;
- [ ] runner, snapshot, memory, receipt, campaign ordering, and validator;
- [ ] existing P0/AL route/certificate/runner compatibility; and
- [ ] strict-parity/default-route regression boundaries.

Required post-flight checks are Ruff, formatting, scoped Pyright, compileall,
`git diff --check`, source/runtime identity, dirty-tree scope, frozen hashes,
active-process audit, and independent review. Test counts alone never establish
scientific or performance completion.

## API evolution and rollback

Observable changes are additive: new opt-in SQP preparation/result APIs, a new
route identity, new v2 campaign schemas, and additional explicit CLI dispatch.
Existing AL numerical behavior, result schemas, paths, enum interpretation in
legacy v1 receipts, and default solver selection must remain unchanged.

Before implementation completion:

- [ ] inventory every caller of the changed fullspace route and receipt APIs;
- [ ] add compatibility tests for v1 bytes, route order, errors, and AL output;
- [ ] document that existing callers require no migration;
- [ ] keep v1 readers indefinitely; no deprecation is authorized; and
- [ ] prove rollback by removing SQP registration while legacy tests and
  artifact validation continue to pass.

## Risks and mitigations

- **Reverse-row tape memory:** fixed batch width 8, exact-tail batches, whole-
  child memory monitor, and immediate kill at 80% device memory.
- **Constraint rank deficiency:** diagnostic SVD gate plus fail-closed Schur
  Cholesky/KKT residual; never remove constraints or use a silent pseudoinverse.
- **Poor BFGS curvature:** Powell damping, explicit SPD symmetry, recorded
  identity reset, and bounded consecutive-reset failure.
- **Merit stagnation:** exact l1 merit, multiplier-dominating penalty, bounded
  device Armijo, one frozen reset retry, then fail closed.
- **False solver convergence:** independently recompute raw physical KKT,
  feasibility, objective, and all endpoint observables.
- **Schema drift invalidates AL history:** explicit legacy-v1 route list,
  golden hashes, additive v2 schemas, and schema-dispatched validator.
- **Compile or runtime exceeds native:** one-step/ten-step measured-path kill
  before cold; compile reported separately and never hidden.
- **A fast invalid endpoint is promoted:** raw results are non-certifying and
  warm execution is impossible before a complete cold certificate.
- **Dirty-tree contamination:** explicit intended source manifest and immutable
  per-run snapshots; unrelated work remains untouched.

## Completion criteria

### Engineering speed achieved

- [ ] Physics, equations, 716 DOFs, 255 equalities, fixed state, FP64 authority,
  and certificate tolerances are independently proven unchanged.
- [ ] The one-dispatch hot solve has zero callbacks and zero hot transfers.
- [ ] Cold and all three warm endpoints independently pass objective,
  feasibility, raw KKT, finite-state, projection, branch, cross-evaluator,
  field-line, fixed-state, and provenance gates.
- [ ] All three warm synchronized solve times are below
  `287.30421751597896 s`; median/range and cold compile/total times are reported.
- [ ] The result is labelled an engineering achievement while the formal
  comparative verdict remains `NOT_PRODUCED` absent a matched baseline.
- [ ] Production promotion is not implied; any later promotion is explicit and
  reversible.

### Bounded-negative complete

- [ ] `CFS-SQP1` has an executed terminal disposition bound to immutable raw
  evidence, or an earlier `NOT_SELECTED_BY_GATE` record bound to its decisive
  artifact.
- [ ] The evidence identifies the exact correctness, rank, KKT solve,
  convergence, memory, transfer, compile, or projected-time failure.
- [ ] No partial, invalid, projected, or utilization-only evidence is described
  as a speed verdict.
- [ ] Prior AL artifacts and production/default routing remain unchanged.
- [ ] The closure does not claim that all SQP, primal-dual, or GPU formulations
  are impossible.

Either terminal section satisfies execution of this SSOT. Only the first
section satisfies the engineering speed objective.

## Open questions

None before implementation. Any evidence requiring a change to batch width,
regularization ladder, merit rule, BFGS rule, iteration budget, or endpoint
gate creates a new route identity and requires an explicit dated SSOT revision
before execution.
