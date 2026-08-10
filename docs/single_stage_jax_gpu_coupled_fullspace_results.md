# GPU-Native Coupled/Fullspace Single-Stage JAX Results

**Status:** `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`; GPU-native execution is faster than the native threshold, but neither complete constrained route satisfies the objective, feasibility, or KKT gates; formal comparative verdict `NOT_PRODUCED`
**Execution SSOT:** `docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md`
**Phase-0 budget artifact:** `docs/single_stage_jax_gpu_coupled_fullspace_phase0_budget.json`
**Implementation start:** 2026-08-09

## Current disposition

The source-bound physics ledger, 716-coordinate layout, finite route knobs,
canonical receipt primitives, validator, runner CLI preflight, immutable source
snapshot/runtime evidence producer, pure explicit-state evaluator, bootstrap
adapter, and canonical bootstrap-artifact contract now exist. Matrix-free
constraint JVP/VJP products and direct fullspace feasibility/KKT primitives are
also implemented. Focused tests currently pass, including transfer-guarded
shared-helper checks. The authoritative immutable RTX 5090 bootstrap and
same-state parity campaign completed at
`/home/jungdaesuh/campaigns/simsopt-fullspace-bootstrap-phase1-r4`. Its sealed
completion, bootstrap, runtime-evidence, and manifest SHA-256 values are,
respectively,
`7bf43ed3d9398ae05efefeced5a5b30de9fb107b88fd1a11769dabe065614983`,
`317255cd183081912f130329e957deebda5a5c1106353760ec58112f0601cee7`,
`439250a3b1036c23769f8d1614eee3b15b55946fa0af58cd2d313473ea01bf43`,
and `22c4c9697b4bab9c2491eb023255c07534e931d768b93ae8d6872131f3c85b35`.
All four artifacts are read-only. The initial exact Boozer residual norm is
`6.405274698838223e-14`, and all 17 same-state fields match the authoritative
evaluator with zero reported difference at state SHA-256
`9dd00baea8fc7362b7df76d9d927f270d61203eb5b3a4f1aa95e7338e2a44da8`.
The independent campaign validator passes the sealed completion record.

The first provenance-bound coupled/fullspace derivative execution passed on the
RTX 5090 at
`/home/jungdaesuh/campaigns/simsopt-fullspace-first-eval-rtx5090-r2`.
The combined value, gradient, changed-state constraint JVP, and constraint VJP
compiled in `20.036098747 s` and executed synchronously in `0.037187907 s`.
The value and derivatives were finite, the JVP/VJP transpose relative error was
`2.426538461498695e-16`, and the guarded hot path recorded zero D2H transfers.
The sealed first-eval artifact SHA-256 is
`712633553f0001e60d7d106f5980b4751cbcc5b4439d15a7c6fbe7100d53a250`.
This is a derivative-graph timing, not an optimizer or end-to-end speedup; no
speed verdict or terminal disposition is currently available.

The separate Landau A100 first-evaluation gate passed at
`/home/jungdaesuh/campaigns/simsopt-fullspace-first-eval-a100-r2`, using the
Landau-compatible native extension sealed into its own immutable snapshot. The
combined derivative graph compiled in `52.091429185 s` and executed
synchronously in `0.052973222 s`; every reported value was finite, the JVP/VJP
transpose error was exactly zero, and the guarded hot path recorded zero D2H
transfers. The first-eval artifact SHA-256 is
`616a915cca65834e7576dadffab674aa1b9fc257c850158ae0e7f20474ac63cd`.
The RTX and A100 timings remain separate device/runtime observations and are not
pooled or treated as a cross-host performance ratio.

The public callback-free fused L-BFGS seam and the `CFS-P0` fullspace state
machine are implemented. `CFS-P0` uses centered dimensionless
coordinates about the bootstrap and the frozen merit
`M0 = Phi + (10/2)||Dq||_2^2`. Its objective, equality vector, diagnostics, and
scaled gradient share the explicit-state evaluator rather than invoking the
nested Boozer solve or implicit adjoint. The prepared optimizer retains its
fixed-shape loop and state on device; its public result exposes only typed JAX
arrays. CPU reference tests currently pass for centered-coordinate round trips,
the exact merit formula, scaled-gradient finite differences, public solver
reuse, and a 10-step fullspace integration with strict merit and scaled-
feasibility progress. The GPU transfer/progress results below supersede this
CPU-only implementation gate.

The formal comparative campaign verdict remains `NOT_PRODUCED`. The historical
native timing is an engineering route-selection threshold because the
claim-compatible native baseline receipt is absent.

The first provenance-bound 10-step `CFS-P0` solver canary passed on the RTX
5090 at
`/home/jungdaesuh/campaigns/simsopt-fullspace-cfs-p0-canary-rtx5090-r1`.
Preparation and compilation took `145.313756532 s`; the synchronized 10-step
device solve took `0.242000841 s` with 14 value/gradient evaluations. `M0`
decreased from `9.838666131002996e-05` to `8.478396370445579e-05`, scaled
feasibility decreased from `2.3915936023119927e-04` to
`1.221691355593912e-04`, and scaled stationarity decreased from
`1.6405465754011186e-01` to `1.5935718534369361e-03`. The cumulative
nonfinite-evaluation count was zero, every accepted state and the endpoint were
finite, and the guarded solve recorded zero hot D2H transfers. The canary
artifact SHA-256 is
`0a97b54d31cab9cea015f804ee7f7efd05274c071aa135896177dd19218ed783`.
Linear timing extrapolation is approximately `24.20 s` per 1000 inner
iterations, but that is only a route-selection projection; it excludes
augmented-Lagrangian outer work, endpoint certification, and complete-solve
behavior and therefore is not a speed result.

The matching Landau A100 child was launched from immutable manifest
`1a02b4db63a9525f7cf22c6338ce2c66f229a03f0434c11d1b6c8bb6e3f46009`
at `/home/jungdaesuh/campaigns/simsopt-fullspace-cfs-p0-canary-a100-r1`.
It published runtime and bootstrap evidence but did not publish a canary
artifact before the frozen 180-second containing-process timeout. Its route
disposition is `NOT_PRODUCED_PROCESS_TIMEOUT`, not a performance loss or speed
measurement. No child process remains. The RTX lane is therefore the selected
device for the remaining solver work; the A100 run will not be replayed merely
to obtain a timing.

The provenance-bound RTX 5090 100-step `CFS-P0` canary then passed at
`/home/jungdaesuh/campaigns/simsopt-fullspace-cfs-p0-canary-100-rtx5090-r1`.
Preparation/compilation took `29.884938623 s`, and the synchronized 100-step
solve took `1.257042403 s` for 109 value/gradient evaluations. `M0` decreased
from `9.838666131002996e-05` to `7.972151297700921e-05`; scaled feasibility
decreased from `2.3915936023119927e-04` to `2.341379990715449e-04`; all state
was finite, the nonfinite-evaluation count was zero, and hot D2H remained zero.
The canary artifact SHA-256 is
`b66eed04c372cc2d8c02649c9cc7d60b449c96797e1c63f49325c1f0e3d2fb85`.
A linear inner-iteration projection is approximately `12.57 s` per 1000
iterations, far below the historical native threshold, but it remains
non-promoting because `CFS-P0` is a fixed-penalty diagnostic and does not
certify equality feasibility or KKT conditions. The small net feasibility
improvement after 100 steps, compared with the larger early improvement after
10 steps, is direct evidence that the promoting augmented-Lagrangian route is
required rather than treating this penalty endpoint as a solution.

The complete fixed-shape `CFS-AL1` device program is now implemented. It runs
ten augmented-Lagrangian stages in one compiled `lax.scan`; each stage invokes
a fresh 100-step parametric fused L-BFGS solve, updates the multiplier and
bounded penalty on device, and resets quasi-Newton history. The final result
reports post-update raw multipliers, raw KKT stationarity, scaled feasibility,
stage history, cumulative evaluation counts, and finite-state latches. A
focused CPU execution of the complete ten-stage program passes, as do the
scaled/raw multiplier identity and penalty-update tests. Ruff, formatting,
Pyright, and compileall pass for the solver and public parametric optimizer
seam. The formulation-specific endpoint certificate contract and provenance-
bound GPU runner subsequently passed their focused CPU gates before execution.

The first immutable complete `CFS-AL1` RTX 5090 execution is sealed at
`/home/jungdaesuh/campaigns/simsopt-fullspace-cfs-al1-complete-cold-rtx5090-r1`.
Its synchronized ten-stage, 1000-inner-iteration device solve took
`13.829276024 s` after `234.700459232 s` of preparation and compilation, with
zero hot D2H transfers. All ten stages completed, all 1103 value/gradient
evaluations and accepted states were finite, and the result artifact SHA-256 is
`617bc9ca71e17da2d64ff67358a1953ce96b885ea4e5a7f67687ccf5556cf1f0`.
The result is read-only and bound to immutable snapshot manifest
`2c78732f6dbccf11bffa633bbbe85e1dea450d5e3293ac97487c3e1ff56f3f3f`.

The route did **not** converge: final scaled feasibility was
`1.316019502709744e-05` against `1e-10`, raw KKT stationarity was
`1.0417891670331875e-01` against `1e-7`, and physical objective was
`7.33698805736055e-05` against `4.4822247e-08`. Therefore its terminal status
is `SOLVER_RESULT_NOT_CONVERGED`, `promotion_eligible` is false, and no endpoint
certificate was produced. This result establishes that host/device traffic is
no longer the active bottleneck for this route; the remaining failure is
optimizer convergence under the frozen AL schedule. Warm repetitions of the
same failing route are not justified.

The evidence-selected `CFS-AL2` accuracy escalation retained the identical
physics, scaling, multiplier update, penalty sequence, ten-stage device scan,
and certificate tolerances while increasing only the inner limit from 100 to
1000 iterations per stage. Its one authorized immutable RTX 5090 cold sample
is sealed at
`/home/jungdaesuh/campaigns/simsopt-fullspace-cfs-al2-complete-cold-rtx5090-r1`.
The synchronized 10000-inner-iteration solve took `132.722050532 s`, which is
`2.164` times faster than the `287.30421751597896 s` native engineering
threshold. Preparation and compilation took `226.471364192 s`; the hot solve
recorded zero D2H transfers. All ten stages, 10708 value/gradient evaluations,
and accepted states were finite. The read-only result SHA-256 is
`1eb3998aea4e3f64e0a0e328b03afca6fee022d32bebaca78ca1f9e3051bff53`,
bound to immutable snapshot manifest
`03d2c5ca3c44969ea79925f6a75588225bb2ea1f5b2364f01a6bfea35a443233`.

`CFS-AL2` also did **not** converge. Final scaled feasibility was
`4.084586796758132e-06` against `1e-10`, raw KKT stationarity was
`5.938484797143023e-02` against `1e-7`, and physical objective was
`2.0986632580462576e-05` against `4.4822247e-08`. Every inner solve again ended
at its iteration limit; final augmented stationarity was
`1.6218350722980084e-03` against `1e-7`. A further tenfold L-BFGS budget would
project to approximately `1327 s`, already about 4.6 times slower than native,
before endpoint audits. It is therefore rejected by the measured-path kill
rule. `CFS-AL1-B` is `NOT_SELECTED_BY_GATE`: no >=20% globalization-
fragmentation evidence exists, and the observed line-search evaluation excess
does not identify globalization as the dominant failure.

The engineering conclusion is `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING` for
the authorized L-BFGS augmented-Lagrangian route family. The work did establish
the intended GPU architecture—one device-resident dispatch, zero hot D2H, and a
sub-native 10000-iteration solve boundary—but fast execution is not a valid
solution without the frozen objective, feasibility, KKT, branch, and endpoint
certificate gates. No warm repetitions or endpoint audits are run for a
nonconverged endpoint. A future tranche must use a stronger constrained method,
such as a GPU-native SQP or primal-dual formulation, under a new SSOT revision;
it must not reinterpret these failed routes as a speed win.

| Route | Terminal disposition | Decisive evidence |
| --- | --- | --- |
| `CFS-P0` | `DIAGNOSTIC_EXECUTED / NON_PROMOTING` | RTX 10/100-step artifacts pass finite, progress, and zero-hot-D2H gates but fixed-penalty P0 cannot certify KKT |
| `CFS-AL1` | `SOLVER_RESULT_NOT_CONVERGED` | Read-only artifact `617bc9ca...` misses objective, feasibility, and KKT gates after 1000 iterations |
| `CFS-AL1-B` | `NOT_SELECTED_BY_GATE` | Required >=20% globalization-fragmentation evidence is absent; evaluation excess is only 8--13 per 100 inner iterations |
| `CFS-AL2` | `SOLVER_RESULT_NOT_CONVERGED` | Read-only artifact `1eb3998a...` is fast but misses objective, feasibility, and KKT gates after 10000 iterations |

The strict-parity example and existing production/default route were not
changed or promoted. The new fullspace routes remain explicit opt-in APIs and
benchmark machinery. Existing unrelated worktree changes, including the
pre-existing tracked `src/simsopt/geo/boozersurface.py` modification, remain
untouched.

## Implementation-start source record

The implementation began from the following live worktree identity:

| Record | Value |
| --- | --- |
| Git `HEAD` | `320e5cba814414a43e48cb5b6e53f4ad356a9925` |
| Tracked-diff SHA-256 | `a78204a047663527a2ad3a5ea50defc2557c06fe30920c079b5cbfd733a487d6` |
| Untracked path-list SHA-256 | `adb87106cec8259c3070e0a3f1d56c97677c240478e0ee8a48b0b0d58de85d10` |
| Plan SHA-256 | `bc10db29c5ed34a54c1667b3417a0e8e08ffbd870124f66792ab446bca669c3f` |

The untracked digest binds only the sorted untracked **path list** at the
implementation boundary. It is not a byte or content manifest. Claim-bearing
runs must use the later immutable snapshot manifest, which hashes the actual
executed bytes and binds them to runtime imports.

The worktree was already dirty and contained unrelated user-owned tracked and
untracked work. This tranche must preserve that work and scope every later
source snapshot and diff to the intended fullspace files.

## Design classification and design-it-twice decision

This is a **Tier 3** change: it adds a cross-module public solver route,
scientific contracts, receipts, and promotion boundaries. It therefore
requires explicit interfaces, fail-closed API evolution, integration tests,
and the complete post-flight checks in the implementation plan.

Two solver-boundary designs were considered:

1. Import the legacy private `fused_stepwise` implementation directly from the
   fullspace solver. This is smaller, but makes a private optimizer API a public
   cross-module dependency.
2. Promote a narrow public, typed, callback-free seam over the existing
   fixed-shape fused L-BFGS kernel, then make the dedicated fullspace solver
   depend only on that seam. Keep fullspace physics, scaling, route policy, and
   later augmented-Lagrangian state in the fullspace module.

Design 2 is selected and implemented for `CFS-P0` and `CFS-AL1`. Public fullspace consumers
do not import private optimizer modules, the hot optimizer loop has no callback
or host materialization, and the transfer boundary can be audited explicitly.
The coupled outer state machine uses the public parametric seam and remains one
fixed-shape device dispatch.

Two contract-ownership designs were also considered:

1. A monolithic benchmark JSON contract duplicated by the physics and solver
   implementations.
2. Domain-owned immutable contracts: the objective core owns the physics
   ledger, layout, and objective configuration; the solver owns its tagged
   route policy and options; the receipt serializes and cryptographically binds
   their canonical payloads.

Design 2 is selected because it keeps each rule at its authoritative source and
prevents the benchmark layer from becoming a second definition of the physics
or algorithm.

## Frozen historical gap and engineering budgets

The stopped nested GPU route took `7541.455 s`; the fastest historical native
engineering trajectory took `287.30421751597896 s`. Therefore the coupled
route must close

```text
7541.455 / 287.30421751597896 = 26.24902295275414x
```

or approximately **26.25x**, while independently reaching objective
`<= 4.4822247e-08` and passing the full feasibility and KKT certificate. The
preferred 10% margin is `<= 258.57379576438106 s` for every warm solve.

The machine-readable Phase-0 budget freezes these planned limits:

- 10-step canary containing-process timeout: `180 s`;
- 100-step canary containing-process timeout: `360 s`;
- complete warm solve-boundary timeout: `360 s`;
- complete containing-process timeout: `900 s`;
- peak GPU memory: at most `0.8` of physical device memory; and
- transfer boundary: one initial H2D, zero hot-loop D2H, and one final D2H.

The budget artifact labels every graph-reduction entry
`PLANNED_NOT_MEASURED`/`NOT_MEASURED`. These are architectural hypotheses, not
profile evidence or projected speedups. It deliberately contains no dynamic
bootstrap-derived targets.

## Phase ledger

| Phase | Status | Evidence required before completion |
| --- | --- | --- |
| 0 — contract and evidence boundary | **COMPLETE** | Ledger/routes/budgets, CLI, immutable snapshot/runtime evidence, canonical receipts, validator, synthetic campaign gates, and real bootstrap publication pass |
| 1 — pure joint state and physics | **COMPLETE** | The sealed r4 real bootstrap and completion record pass independent validation; 17/17 authoritative same-state comparisons are exact |
| 2 — derivatives and certificates | **CLOSED_INCOMPLETE** | Derivative and fail-closed certificate primitives pass; cross-evaluator/field-line execution is correctly skipped because no certifiable endpoint exists |
| 3 — device-resident solver | **COMPLETE** | Public fused/parametric seams and P0/AL1/AL2 device programs pass; sealed RTX solves have zero hot D2H |
| 4–5 — GPU graph work and canaries | **COMPLETE** | P0, AL1, and AL2 evidence isolates convergence rather than transfers or device execution as the limiting factor; AL1-B is not selected |
| 6 — complete certified solve | **REJECTED_BY_GATE** | AL1 and AL2 reach frozen termination without objective, feasibility, or KKT convergence; warm repeats and endpoint audits are prohibited |
| 7 — terminal disposition | **COMPLETE** | `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`; formal comparative verdict `NOT_PRODUCED` |

The focused consolidated gate currently reports `149 passed` across the
fullspace contract, core, derivatives, bootstrap adapter/artifact/runner,
same-state parity, route, snapshot/runtime, campaign/validator, and selected
single-stage integration suites.

After the solver changes, 64 focused fullspace/runner tests and 147 shared
L-BFGS parity/regression tests pass in the pinned CPU reference environment.
The final AL2 integration gate adds 30 combined route/core/certificate tests
and 2 focused runner/publisher tests; Ruff, formatting, scoped Pyright,
compileall, and `git diff --check` pass.

## Result boundary

The GPU-native graph achieved the intended execution result, including a
`132.722050532 s` synchronized 10000-iteration boundary and zero hot D2H, but
did not produce a scientifically valid endpoint. Therefore there is no speed
win claim. The engineering route family is closed bounded-negative and the
formal comparative protocol verdict remains `NOT_PRODUCED`.
