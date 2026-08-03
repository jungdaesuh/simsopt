# Custom JAX Quasi-Newton Performance and Boozer Closure Plan

**Status:** In progress — diagnosis and architecture decision complete
**Last updated:** 2026-08-02
**Change tier:** Tier 3 — solver behavior, routing, schemas, and scientific evidence

## Purpose

Close two independent gaps left after the custom JAX BFGS/L-BFGS prepared
runtime refactor:

1. make the host-driven Boozer outer BFGS workflow report and reach a certified
   native/JAX endpoint; and
2. determine whether a production-routed custom L-BFGS fast path can reduce the
   measured `coil47` overhead without changing the parity path.

The completed implementation history remains in
`docs/jax_custom_bfgs_lbfgs_optax_structure_implementation_plan.md`. This file
owns only the follow-on closure work.

## Goals

- Produce a certified Boozer outer endpoint on native CPU, JAX CPU, RTX 5090,
  and A100 lanes.
- Preserve the fixed exact native-scale inner solve and its FP64 CPU/GPU
  accuracy.
- Make Boozer example and receipt success fail closed on outer convergence.
- Measure and, if viable, reduce custom L-BFGS warm overhead on `coil47` while
  leaving the native-compatible parity route unchanged.
- Report preparation, first execution, warm execution, host RSS, device memory,
  program count, and StableHLO size on matched boundaries.
- Produce semantically validated, clean-checkout, off-host-replayable evidence.
- Preserve authentic RED -> GREEN -> REFACTOR evidence for every new behavior
  change.

## Non-Goals

- Do not replace custom solvers with Optax or make Optax the parity oracle.
- Do not require identical custom, Optax, and SciPy trajectories.
- Do not weaken inner-solve, gradient, endpoint, or placement tolerances.
- Do not accept a capped, line-search-failed, or nonstationary zero-step Boozer
  endpoint as converged. A zero-step result remains valid only when the initial
  point already passes the canonical stationarity predicate and the provider
  reports success.
- Do not JIT the complete eager outer optimization loop.
- Do not add a large benchmark matrix; use Boozer for BFGS correctness and
  `coil47` for L-BFGS performance.
- Do not recreate historical RED revisions for completed work.

## Current Context

### Confirmed facts

- Commits `19194b957` and `52b543aec` fix and receipt the native-scale exact
  Boozer inner solve. Commit `3adaf554b` updates the receipt count.
- The 255-dimensional exact system previously used GMRES restart `64`. The
  exact path now permits restart `256`; the generic policy is unchanged.
- Inner CPU/GPU residuals are approximately `2e-29`, with RMS errors around
  `6e-15` to `7e-15`. The diagnostic receipt is
  `docs/receipts/custom-quasi-newton/boozer-exact-inner-native-scale-cpu-gpu-20260802/`.
- Both recorded parity outer BFGS runs accepted zero steps and stopped after
  six evaluations. They are not convergence evidence.
- The Boozer example currently computes top-level `status="ok"` from finite,
  decreasing scientific observables without requiring
  `optimizer_result.converged`. The same payload separately records
  `outer_solver_success=false`; this is a production finalization defect.
- The clean RTX 5090 `coil47` receipt is
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-five-sample-vram-20260802/`:

  | Metric | Custom JAX | Optax | Interpretation |
  | --- | ---: | ---: | --- |
  | Cold | 14.43294 s | 4.07767 s | Custom is 3.54x slower |
  | Warm | 0.044042 s | 0.027639 s | Custom is 1.59x slower |
  | Iterations | 12-13 | 15 | Iteration count is not the slowdown |
  | Final objective difference | 5.55e-17 | reference | Equivalent endpoint |
  | Final gradient infinity norm | 5.87e-10 | 1.82e-11 | Both pass 1e-8 |
  | Median VRAM | 1480 MiB | 1472 MiB | Typical use is tied |
  | Maximum VRAM | 1514 MiB | 2602 MiB | Optax maximum contains one outlier |
  | Process RSS | 1,786,248 KiB | 1,605,436 KiB | Custom uses about 177 MiB more |

- Absolute peak RSS hides part of the provider cost: the same receipt has
  maximum solver-attributable RSS deltas of `386,412 KiB` custom and
  `203,912 KiB` Optax, a `1.895x` ratio.
- Custom L-BFGS preparation compiles initialization, value/gradient, three
  reverse-communication entry kernels, and result extraction. Optax prepares
  two main programs in the benchmark comparator.
- A no-callback custom L-BFGS warm run records two `advance` host observations
  per accepted iteration (`24/12` or `26/13`). The opportunity is specifically
  to fuse `NEW_X` reentry with the next accepted-step search, not to recreate a
  whole-solve or all-entry generic graph.
- `--intent fast` currently selects an execution profile only. It does not
  select a different private L-BFGS implementation. Production L-BFGS routing
  is controlled by `lbfgs_run_mode`; the Boozer example uses the separate
  host-core BFGS/L-BFGS implementation.
- `benchmarks/lbfgs_ondevice_compile_shape.py` currently covers custom private
  L-BFGS diagnostics, not the exact custom/Optax provider factory pair.
- The current receipt validator verifies bytes, environment-lock hashes, and
  archive parity. It does not recompute scientific or performance gates.
- The exact-inner adversarial review and scoped audit pass. Overall promotion
  remains open.

### Working hypotheses

- The Boozer line search may encounter candidate inner-solve failures. The
  objective converts a failed candidate into a discontinuous rejected value,
  which can prevent Wolfe acceptance. Trial-level evidence is still missing.

### Confirmed warm-gap attribution (2026-08-02 RTX 5090 profiling)

- The `NEW_X` reentry hypothesis is refuted as the dominant cause: only
  about `35%` of the custom-minus-Optax warm delta is host-boundary time;
  about `65%` is device compute. Custom executes `14,394` GPU ops and
  `15.5 ms` device time per warm solve versus Optax `3,062` ops and
  `4.8 ms`, with `1,477` versus `78` in-program while-predicate D2H reads.
- The stepwise driver rebuilt `_int_scalar(maxiter/maxfun)` every macro step
  (52 H2D per solve); hoisted trajectory-identically in commit `7d88bd1a3`.
- Projections under the receipt environment: scalar hoist alone reaches
  `1.36-1.53x`; adding `NEW_X` reentry fusion reaches `1.10-1.28x` — not
  par; only a fully on-device driver or an equivalent device-side op
  reduction reaches `0.85-1.02x`. The par target therefore requires more
  than the originally planned `fused_stepwise` kernel; the revised Phase 6
  evaluates a bounded on-device fast route (measuring the existing
  `monolithic_debug` program first) against device-side loop restructuring,
  and records that decision here before implementation.
- Measured 2026-08-02: `monolithic_debug` is not a viable fast route — its
  `lbfgsb_mainlb` path omits `unconstrained_fast_path=True`, compiling the
  full bound-constrained machinery (`462.5 MB` StableHLO, `491 s` to lower,
  compile unfinished at 21 minutes and 20 GiB; fails both watchdog bounds).
  The selected Phase 6 design is instead the bounded fused on-device
  driver: one `lax.while_loop` whose body is a `lax.cond` over the existing
  search/`NEW_X`-reentry transitions with `unconstrained_fast_path=True` —
  the current host driver moved on-device. The measured prototype compiles
  in `12-17 s` / `1.8 GiB`, removes all per-step host observations, is
  endpoint-identical to stepwise at matched iterations (`|df| = 0.0`,
  `max|dx| = 1.6e-14`), and projects `0.85-0.95x` Optax warm under the
  receipt environment. The earlier blanket rejection of an all-entry
  on-device graph was premised on `monolithic_debug`'s compile cost and is
  refuted for the fast-path-flagged variant. Callback-bearing callers stay
  on `stepwise`; the fused route rejects observers fail-closed. Durable-par
  caveat: custom device time remains `2.8x` Optax's; the dominant device
  costs are `~1,513` while-predicate D2H per solve and the `formk`/`bmv`
  subspace `fori_loop` fusion family.
- Caveat: par is defined against Optax's current host-taxed warm median
  (`27.6 ms`, platform allocator). Custom's device floor (`15.5 ms`) exceeds
  Optax's (`4.8 ms`); device-side excess must shrink for durable parity.

### Confirmed outer-failure diagnosis

- Matched native/custom diagnostics agree through the early and middle BFGS
  trajectory. At 20 iterations, both use 23 evaluations and final objectives
  differ by `5.86e-17`.
- The baseline-anchored custom run reaches iteration 789, then its exact
  predictor crosses to a different Boozer branch for ordinary line-search
  displacements. The line search eventually operates at numerical-noise step
  lengths and fails.
- The local implicit gradient is correct: central differences from `1e-10` to
  `1e-7` agree with the analytic directional derivative.
- Newton initialized directly from the last accepted inner state succeeds for
  tested displacements from `1e-6` through `1e-3`, with residual norms near
  `1e-14`.
- A full accepted-incumbent prototype completes 1,000 accepted BFGS steps with
  no inner failures and reproduces the native capped trajectory: 1,187 custom
  evaluations versus 1,188 native, and final objectives `3.72242e-7` versus
  `3.71939e-7`. This fixes the premature line-search failure but does not by
  itself prove convergence.

## Rationale

The Boozer BFGS correctness failure and the L-BFGS performance gap have
different owners and fixtures. Keeping them as separate workstreams prevents a
fast-path benchmark change from being mistaken for a Boozer scientific fix.

Boozer-specific continuation belongs in the objective if the trace confirms an
inner predictor or Newton basin failure. A generic optimizer-level rejection
protocol is the fallback only if bounded objective-owned recovery cannot return
a certified value.

The existing L-BFGS parity route remains the compatibility baseline. A fused
route is eligible only as a separately identified fast implementation whose
receipt proves that it actually ran.

## Assumptions

- Native SciPy/SIMSOPT remains the parity oracle.
- Optax remains an explicit comparator, never parity authority.
- Existing CPU and GPU lock files remain the environment authority.
- RTX 5090 and A100 execution can use isolated FP64 environments with strict
  device selection.
- `src/simsopt_jax/parity_tolerances.py` remains the only numerical-tolerance
  owner; this plan does not duplicate or silently revise its values.

## Scientific Certificate

Promotion uses the existing tolerance SSOT:

- initial objective and gradient:
  `mirror_single_stage_initial_objective` and
  `mirror_single_stage_initial_gradient`;
- final value and parameters:
  `mirror_single_stage_final_value` and
  `mirror_single_stage_final_parameters`;
- terminal gradient infinity norm:
  `native_workflow.terminal_stationarity_atol` (`1e-7`);
- geometry invariants: `mirror_surface_invariant`; and
- constraints, when present: `native_workflow.terminal_constraint_norm_atol`.

Every promoted lane must report `success=true`, raw provider success, and
`stopping_reason="converged"`. It must have at least one accepted step unless
the initial gradient already satisfies the terminal stationarity tolerance and
the raw provider reports convergence at iteration zero. Iteration-limited,
line-search-failed, nonfinite, or merely decreasing endpoints fail.

Iterations and evaluations are recorded but may differ by provider. Raw
parameters remain available even when an invariant representation is the
scientific comparison owner.

### Matched-budget equivalence milestone (2026-08-03 author ruling)

At a matched 1000-iteration budget neither implementation reaches the
`1e-7` terminal stationarity above: the native SciPy oracle itself caps at a
gradient infinity norm of `9.2e-4` (custom reaches `4.6e-5`). The plan
author's adversarial final review therefore ruled: record **matched-budget
equivalence versus the native oracle** as a separately named
interoperability/performance milestone — with explicit endpoint, objective,
gradient, parameter, counter, and status criteria and distributional GPU
evidence — while the scientific certificate above remains the unmodified
promotion gate and is explicitly **not yet achieved** by either
implementation. Matched-budget receipts must be labeled as such and cannot be
presented as scientific-certificate promotion.

## Design Alternatives

### Boozer candidate recovery

1. **Objective-owned bounded recovery — preferred when supported by the trace.**
   Keep Boozer continuation knowledge in
   `surface_objectives_traceable.py`. Try one fixed, bounded sequence of damped
   predictor or baseline-seeded inner solves and return only a certified
   physical value/gradient.
2. **Optimizer-owned rejected-evaluation protocol.** Add one typed private
   evaluation outcome that tells the line search to contract the step without
   treating a synthetic penalty as a physical objective.

Select option 1 only for predictor or inner-solve basin failure. Select option
2 only if certified candidate domain holes remain after bounded recovery. Do
not implement both speculatively.

**Selected design:** objective-owned accepted-incumbent continuation. The
objective exposes an immutable inner state containing accepted coil DOFs,
solved inner DOFs, raw physical objective, and eligibility. Every speculative
trial receives the same explicit state. A candidate returns a new immutable
state, which the host driver promotes only from its post-acceptance callback.
If the exact linear predictor fails, the objective makes one bounded retry by
running Newton directly from the incumbent inner DOFs. Only a certified primal
and adjoint result is eligible for promotion.

**Rejected design:** session-owned or `BoozerSurfaceJAX`-owned mutable
continuation. It would let rejected trials contaminate subsequent evaluations,
couple correctness to evaluation order, and make concurrent sessions unsafe.
Promotion from the line-search observer is also rejected because that observer
runs before acceptance is decided. The generic optimizer-owned rejection
protocol remains unnecessary unless the bounded incumbent retry still leaves
certified-domain holes.

### L-BFGS fast routing

1. **Current `stepwise` parity path.** Retain `start`, `search`, and
   `new_x_reentry` kernels and SciPy-compatible behavior.
2. **Explicit `fused_stepwise` fast path.** Add a prepared, bounded
   `NEW_X`-reentry-plus-search kernel beside `PreparedLBFGS` in
   `src/simsopt_jax/geo/optimizers/private/_lbfgs.py`. Thread the new typed
   `lbfgs_run_mode` value through
   `src/simsopt_jax/geo/optimizers/optimizer.py` and explicit fast callers.

Parity callers retain `stepwise`; traced callers retain `monolithic_debug`.
The benchmark and application fast lanes must request `fused_stepwise`
explicitly and emit that route in their receipts. No route is inferred from
`--intent` alone, and no public method name changes.

Promote option 2 only if it passes compilation, parity, transfer, and measured
warm gates. Otherwise revert it and retain the measured limitation.

## Schema Evolution Contract

| Object | Current | Planned | Promotion rule |
| --- | ---: | ---: | --- |
| Runner `measurements.json` | 6 | 7 | Version 7 requires solver route, separate preparation/first/warm timings, typed device identity, work/transfer counters, and references to bounded trial or memory traces. Version 6 remains readable as historical diagnostic evidence only. |
| Receipt `manifest.json` and `metrics.json` | 1 | 2 | Version 2 binds runner version, expected sample counts, route, device identity, candidate, lock, archive bundle identity, raw derivations, and qualification verdict. Version 1 remains integrity-checkable but cannot promote this plan. |

`benchmarks/custom_quasi_newton_runtime.py` owns runner version 7.
`benchmarks/custom_quasi_newton_receipts.py` owns receipt version 2 and its
semantic validator. (2026-08-03: the runner advanced to version 8 — version 7
plus trial-trace/GPU-memory production binding, commit 03de61652; the trial
trace advanced to schema 2 adding `final_status`. Version 7 remains readable
as legacy; receipts stay at version 2 with the binding cross-checks.) Tests must cover the current version, the allowed legacy
versions, unknown-version rejection, missing/null/wrong-type fields, nonfinite
metrics, tampered derivations, and historical round trips.

## Implementation Plan

1. Freeze failure, finalization, and schema behavior.
   - [x] Add a RED example regression proving a finite, decreasing Boozer result
         with `outer_solver_success=false` must produce top-level
         `status="failed"`. (65b5d8c9c; smoke contract Option 1.)
   - [x] Add zero-step cases proving stationary raw-success is valid and the
         current nonstationary Boozer zero-step result is invalid.
         (91e1133b9, tests/jax/solve/test_endpoint_certificate.py.)
   - [x] Add a RED runner/receipt test requiring the trial fields needed to
         diagnose line-search failure. (590cffd47, c5c964aeb.)
   - [x] Preserve the exact-inner CPU/GPU test as a GREEN non-regression gate.
   - [x] Implement the version 7/version 2 schema matrix above before producing
         new promotion receipts. (0d8c86ba0, 745fbb850; runner now v8 per note
         above, 03de61652.)
   - [x] Make receipt validation recompute outer and scientific success from
         raw fields; never trust top-level `status` or a stored aggregate
         alone. (745fbb850.)

2. Diagnose the Boozer outer failure with one bounded owner.
   - [x] Add a benchmark-only host diagnostic harness with immutable trial
         records. Correlate optimizer and objective data by
         `(evaluation_index, SHA-256(float64 parameter bytes))`; use no global
         mutable state. (590cffd47, benchmarks/boozer_trial_diagnostic.py.)
   - [x] The host line search owns trial step length and Wolfe errors. The
         solved-pair/objective boundary owns predictor, primal, adjoint, Newton,
         residual, physical objective, and gradient data. The diagnostic
         harness is the only join owner. (590cffd47.)
   - [x] Record predictor kind/success, primal and adjoint success, Newton
         status/iterations, inner residual, raw and filtered objective,
         gradient finiteness/norm, step length, and Wolfe errors. (590cffd47.)
   - [x] Keep capture optional and bounded. Normal execution retains no trial
         trajectory and adds no host callback. (`--capture-boozer-trial-trace`
         opt-in with declared byte caps, c5c964aeb.)
   - [x] Run matched native/custom CPU diagnostics, then the custom strict RTX
         5090 diagnostic, and identify the first causal divergence. (Confirmed
         outer-failure diagnosis section above; GPU deaths are stochastic
         noise-floor line-search mortality, CPU deterministic.)
   - [x] Record the selected recovery design and rejected alternative in this
         plan before changing objective or optimizer behavior. (Selected
         design under Design Alternatives.)

3. Fix Boozer finalization and the diagnosed root cause with RED -> GREEN ->
   REFACTOR.
   - [x] Make the example finalizer require certified outer and inner success,
         finite certificate fields, and the Scientific Certificate above.
         (65b5d8c9c via the endpoint-certificate SSOT, 91e1133b9.)
   - [x] Write an observable RED regression for the selected candidate failure;
         preserve the failing pre-fix command, output, commit, and hash.
         (tests/geo/test_traceable_trial_evaluator.py controller suite,
         1a7ce5b90.)
   - [x] If objective-owned recovery is selected, implement one immutable,
         fixed-length policy in `surface_objectives_traceable.py` and report
         its attempts/evaluations. (Accepted-incumbent continuation with one
         bounded Newton-from-incumbent retry, 1a7ce5b90; lean evaluator core
         5357fc91e; promotion generation guard e3e8d5fe9.)
   - [x] If the optimizer protocol is selected, define one typed private
         outcome and inventory every BFGS/L-BFGS caller before changing it.
         (Not selected — objective-owned recovery chosen; no optimizer
         protocol added.)
   - [x] Require every accepted trial to use a certified inner solve and finite
         physical objective/gradient; never certify a penalty value.
         (`accept` fails closed on uncertified candidates, 1a7ce5b90.)
   - [x] Refactor only after GREEN, keeping candidate classification and retry
         policy under one owner. (5357fc91e single-owner
         `_build_candidate_evaluation_core`.)

4. Close outer Boozer scientific parity.
   - [ ] Use the `boozer` fixture with `method=bfgs`, `maxiter=1000`, matched
         FP64 inputs, line-search settings, and native/JAX budgets. Add an
         explicit bounded Boozer runner watchdog if the current 120-second
         child limit is insufficient; retain the default watchdog elsewhere.
   - [ ] Run native/custom CPU, custom RTX 5090, and custom A100 lanes.
   - [ ] Apply the Scientific Certificate without accepting a cap or failed
         line search.
   - [ ] Compare initial/final components, parameters, invariant geometry,
         gradient norm, constraints, iterations, evaluations, raw status, and
         stopping reason.
   - [ ] Preserve different but equivalent trajectories as such; do not require
         bitwise endpoint identity.

5. Establish matched L-BFGS performance attribution before fusion.
   - [x] Emit fixture-build, provider-preparation, first-execution, and warm
         times as separate synchronized fields. Keep cold total derived.
         (0d8c86ba0.)
   - [x] Give custom and Optax the same work-counter, transfer, RSS, and
         process-attributed VRAM boundaries. Mark genuinely unavailable values
         unavailable rather than inferring them. (0d8c86ba0, e7051ab65.)
   - [x] Extend `benchmarks/lbfgs_ondevice_compile_shape.py` to select custom or
         Optax and compile the exact provider factories used by the runner.
         Record route, per-program and aggregate StableHLO bytes, compile time,
         executable count, dtype, fixture, and options. Share benchmark
         preparation code; do not copy optimizer equations. (6edad18f0.)
   - [x] Profile objective/line-search work, per-kernel device time, and
         host-transfer latency. Confirm or reject the reentry/synchronization
         hypothesis before Phase 6. (REJECTED — warm-gap attribution section
         above: 65% device / 35% host; while-predicate D2H 1,513 vs 78 per
         solve.)
   - [x] Add version-7 boundary tests proving fixture construction and compile
         diagnostics are excluded from warm measurements. (0d8c86ba0.)

6. Evaluate the explicitly routed L-BFGS fast design.
   - [x] Add a RED routing test: `fused_stepwise` must reach the new prepared
         program, while `stepwise` and `monolithic_debug` retain their current
         paths and cache identities. (9cf8c949d, 8fbf50918.)
   - [x] Add a RED transfer test against the measured baseline of two `advance`
         observations per accepted iteration. The fast target is no more than
         `iterations + 1` advance observations, with exact leaves and bytes
         retained in the receipt. (8fbf50918 — fused advance observations
         measure 0 with the transfer gate.)
   - [x] Implement the fused reentry-plus-search program only after Phase 5
         confirms that target is causal. (9cf8c949d, after the Phase-5
         attribution pivot recorded above.)
   - [x] Preserve status, counters, callbacks, zero budgets, `maxfun`, bounds,
         inverse-Hessian behavior, and accepted-state scientific parity.
         (27 fused-vs-stepwise tests, 9cf8c949d; 7 direct fused-vs-SciPy
         contracts, b243a176e; callbacks rejected fail-closed toward
         `stepwise`.)
   - [x] Abort if compilation exceeds 120 seconds or 8 GiB RSS, duplicates
         transition mathematics, changes the endpoint contract, or recreates
         the rejected all-entry generic branch graph. (No abort trips; fused
         reuses the existing `_lbfgsb_scipy` transitions under
         `lax.while_loop`/`lax.cond`.)
   - [x] A performance receipt is eligible only when `intent=fast` and
         `solver_route=fused_stepwise`; the validator rejects any other route.
         (8fbf50918.)
   - [x] Keep the route only if warm custom time is no more than `2.0x` matched
         Optax on both GPUs and improves over the current custom median.
         (RTX 5090: 1.06x quiet / 0.94x contended, from 1.59x pre-fusion;
         A100 superseded — landau down, user directive 2026-08-02 narrows the
         target to the RTX 5090.)

7. Qualify performance, memory, devices, and evidence.
   - [ ] In one prepared child per provider, discard one warm run and retain
         five. Run provider children back-to-back in a seeded AB/BA order on one
         reserved GPU allocation; record clocks, power, utilization, and
         competing GPU processes, and rerun if they materially change.
   - [ ] Record all raw samples and the timing median/range. Gate the maximum
         across retained samples for memory.
   - [ ] Require both maximum absolute process RSS and maximum provider-
         attributable RSS delta (`phase peak - phase start`) to be no more than
         `1.5x` matched Optax.
   - [ ] Require maximum process VRAM no more than `1.5x` matched Optax and a
         same-process prepared warm soak whose RSS, VRAM, and executable count
         plateau rather than grow across retained runs.
   - [ ] Bind GPU model, UUID, compute capability, total VRAM, driver/CUDA
         versions, visible-device selector, and host/job identity. Paired runs
         must use the same UUID and the expected RTX 5090 or A100 model.
   - [x] Repeat the matched `coil47` fast comparison on RTX 5090 and A100 with
         FP64 and no CPU fallback — CLOSED 2026-08-03 after landau returned:
         receipt `coil47-fused-optax-a100-55745feaf` (verdict pass, ratio
         1.1659, five AB/BA rounds, A100-PCIE-40GB identity-bound, all
         endpoints converged in FP64). The regimes invert across GPUs:
         custom fused is 0.55x Optax on the RTX 5090 but 1.17x on the
         A100 (both inside the 2.0x gate) — the fused route's win rides
         on host-tax elimination, and landau's older host/dispatch path
         flips the balance.
   - [ ] Bind every StableHLO and compile artifact to its provider, solver
         route, candidate SHA, and exact device UUID; require a complete
         custom/Optax artifact pair for each GPU.
   - [ ] Publish version-2 manifests and raw artifacts from a clean detached
         candidate checkout.
   - [ ] Add `validate-all --archive-root` so a mounted immutable copy can be
         verified without rewriting receipt manifests. Map by receipt ID,
         verify bundle inventory hash/storage identity, and reject incomplete
         remaps.
   - [x] Copy the authority bundle to a different host or durable store, mount
         it read-only, and replay from a detached checkout at the recorded
         candidate SHA with the recorded environment lock — CLOSED
         2026-08-03 on `landau` (A100 host): 47/47 receipts green; receipt
         `offhost-replay-landau-20260803.md`. Precisely: the VALIDATOR
         checkout was `487d9ff89` (then-HEAD, containing every
         receipt-recorded candidate commit object for authentication);
         the replay re-verified hashes, derivations, qualifications, and
         commit-object presence — no solver execution was replayed at the
         receipts' candidate trees.

8. Complete rollout and API review.
   - [x] Inventory observable behavior changes and every caller of
         `lbfgs_run_mode`, Boozer example finalization, runner schema, and
         receipt schema — receipt `public-routing-inventory.md` (2026-08-01
         snapshot PLUS the 2026-08-03 `fused_stepwise` addendum covering the
         route added after that snapshot) and
         `boozer-compatibility-partitions-20260802.md`; runner/receipt
         schema history in the closure block (schema 9 / trace v3 /
         receipt v2).
   - [x] Document the new `fused_stepwise` option, its explicit fast callers,
         parity preservation, migration path, cache behavior, and rollback —
         `docs/jax_solver_algorithm_matrix.md` "fused_stepwise migration,
         cache, and rollback record (2026-08-03)" (receipt-bound medians)
         plus the rollback receipt `rollback-rehearsal-20260802.md`.
   - [x] Update the solver matrix and provider plan only after gates pass —
         reconciliation commit `a8134ebc8`; medians rebound to the
         receipt-attested `-r2` values on 2026-08-03.
   - [x] Record ordered implementation commits and rehearse rollback in a clean
         worktree — ordered `bb05ec58f..466432b88` series;
         `rollback-rehearsal-20260802.md` proves the pre-fused
         prepared-runtime revert and
         `rollback-rehearsal-fused-20260803.md` proves the fused-route
         lever itself (stepwise fallback, parity green, exactly the two
         route pins failing); clean detached-worktree lane replays are
         attested by the receipts' runner payloads (`git_clean=true`) at
         the successive evidence SHAs `7b1372ad0` and `359fd41fc`.
   - [ ] Obtain independent architecture, numerical, runtime, and evidence
         review with no unresolved finding — IN PROGRESS: two external
         adversarial verdicts (FAIL_ITERATE) received and worked
         finding-by-finding (see the review-iteration addendum); closure
         requires the pending delta review to return PASS.

## Validation Plan

- [ ] Focused strict CPU FP64 tests:

  ```bash
  MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
  .venv-qn-cpu/bin/python -m pytest -q \
    tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
    tests/jax/solve/test_lbfgsb_trajectory_parity.py \
    tests/benchmarks/test_custom_quasi_newton_runtime.py \
    tests/jax/examples/test_single_stage_boozer_vacuum_example.py
  ```

- [ ] Strict CUDA FP64 tests run with a CUDA profile, not the CPU command:

  ```bash
  MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
    tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
    tests/jax/solve/test_lbfgsb_trajectory_parity.py \
    tests/benchmarks/test_custom_quasi_newton_runtime.py \
    tests/jax/examples/test_single_stage_boozer_vacuum_example.py
  ```

- [ ] Matched native/custom CPU Boozer run:

  ```bash
  mkdir -p .artifacts/custom-quasi-newton/boozer-cpu-parity
  MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true SIMSOPT_BACKEND_MODE=jax_cpu_parity \
  SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
  .venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
    --device cpu --intent parity --providers native,custom \
    --cases boozer --method bfgs --maxiter 1000 \
    --output .artifacts/custom-quasi-newton/boozer-cpu-parity
  ```

- [ ] Strict-GPU Boozer uses the same case/options with `--device gpu`,
      `--providers custom`, the strict CUDA environment above, and a receipt-
      validated expected hardware identity.
- [ ] Exact-inner non-regression replays the CPU and CUDA commands recorded in
      `docs/receipts/custom-quasi-newton/boozer-exact-inner-native-scale-cpu-gpu-20260802/summary.md`
      from the detached candidate checkout and rebinds the raw outputs. Only
      inner success, residual, RMS, resolution, and device placement qualify;
      the diagnostic outer status remains non-promotion evidence.
- [ ] `coil47` fast qualification uses `--intent fast`,
      `--providers custom,optax`, `--method lbfgs`, `--maxiter 20`, and requires
      `solver_route=fused_stepwise` for custom.
- [ ] Current local receipt validation remains:

  ```bash
  PYTHONPATH=src:. .venv-qn-cpu/bin/python \
    benchmarks/custom_quasi_newton_receipts.py validate-all \
    --root docs/receipts/custom-quasi-newton --repo-root .
  ```

- [ ] Off-host validation, after the planned archive-root support, runs from a
      clean detached candidate checkout:

  ```bash
  PYTHONPATH=src:. .venv-qn-cpu/bin/python \
    benchmarks/custom_quasi_newton_receipts.py validate-all \
    --root docs/receipts/custom-quasi-newton --repo-root . \
    --archive-root /mnt/immutable-quasi-newton-evidence/receipts
  ```

- [ ] Version-2 semantic validation recomputes sample counts, routes, device
      identity, endpoint certificates, medians/ranges, RSS/VRAM maxima, every
      threshold, and stored verdicts from raw records.
- [ ] Scoped Pyright, Ruff check/format, `compileall`, `git diff --check`, and
      the full project suite pass from a clean checkout.
- [ ] Every new defect has a failing pre-fix run and passing post-fix run bound
      by immutable commit and artifact hashes.

## Risks and Mitigations

- Risk: a finite/decreasing inner result masks outer optimization failure.
  Mitigation: recompute success from raw outer, inner, and scientific fields.
- Risk: a discontinuous penalty corrupts Wolfe reasoning.
  Mitigation: certify physical values only; keep failure classification
  separate.
- Risk: objective-owned recovery hides excessive inner work.
  Mitigation: use a fixed retry bound and report every attempt/evaluation.
- Risk: a fused route is benchmark-only or silently replaces parity.
  Mitigation: use explicit route IDs, caller tests, and fail-closed receipt
  validation.
- Risk: absolute RSS hides doubled incremental memory.
  Mitigation: gate both absolute and provider-attributable RSS.
- Risk: one VRAM outlier or process restart hides growth.
  Mitigation: retain all samples plus a same-process prepared warm soak.
- Risk: GPU claims are attached only to `cuda:0`.
  Mitigation: bind and validate model, UUID, capability, memory, driver, and
  allocation identity.
- Risk: local evidence disappears or cannot be replayed independently.
  Mitigation: validate a read-only off-host copy from a detached checkout.

## Completion Criteria

- [ ] Native CPU, JAX CPU, RTX 5090, and A100 Boozer lanes meet the Scientific
      Certificate. Failed, capped, nonstationary zero-step, and nonfinite
      endpoints remain failed.
- [ ] The exact native-scale inner solve remains green at existing tolerances.
- [ ] The Boozer example and receipt finalizers cannot emit promotion success
      when outer convergence fails.
- [ ] The explicitly identified `fused_stepwise` route meets the `2.0x` Optax
      warm gate on both GPUs and improves over current custom, or it is reverted
      and no performance-promotion claim is made.
- [ ] Absolute and attributable RSS and maximum VRAM meet their `1.5x` gates;
      same-process memory and executable counts plateau.
- [ ] Custom and Optax use matched measurement boundaries, hardware identity,
      and raw five-sample reporting.
- [ ] Cold preparation and first execution are separately reported with exact-
      route StableHLO and executable evidence.
- [ ] No tolerance is weakened and no CPU fallback occurs in strict GPU lanes.
- [ ] RED -> GREEN -> REFACTOR receipts, focused/broad tests, type/style checks,
      clean replay, rollback, semantic validation, and independent review pass.
- [x] Authority artifacts validate from a durable off-host copy
      (`offhost-replay-landau-20260803.md`: landau/A100, detached pristine
      checkout at `487d9ff89`, lock-pinned stack, write-protected archive,
      47/47 green).

## Campaign closure status (2026-08-03)

Forty-four ordered commits in the fixed range `bb05ec58f..466432b88`
(`git rev-list --count`; includes the two external review iterations
recorded in the addendum below — later review-closure commits extend the
series and are recorded there, not in this count); candidate evidence SHA
`359fd41fc`, lanes run from clean detached worktrees, receipts first
committed at `f2ed3d534` and republished under the hardened validator at
`cdf62708a`. Closed with receipts (all `validate-all` green, 47 receipts,
including `--archive-root` against the complete mirror at
`/home/jungdaesuh/qn-receipt-archives`):

- Phase 4 matched-budget milestone:
  `boozer-matched-budget-cpu-triad-359fd41fc` (custom `3.572646605e-7` /
  `4.55e-5` / 1207 evals, bit-identical across four independent runs; native
  `3.719393881e-7` / `9.17e-4` / 1188) and
  `boozer-matched-budget-gpu-distribution-359fd41fc` (five RTX 5090
  repetitions; 2/5 reach the 1000-iteration budget, 3/5 die in stochastic
  noise-floor line search at 920–964; all five endpoints inside the
  3.56–3.77e-7 band containing native). Capped endpoints remain `failed`;
  the Scientific Certificate stays unachieved by both implementations per
  the author ruling above.
- Phase 6/7 performance: `coil47-fused-optax-quiet-359fd41fc-r2`
  (receipt-attested `verdict: pass`, qualification ratio 0.5523 — custom
  warm median 13.14 ms vs Optax 23.79 ms) and
  `coil47-fused-optax-contended-359fd41fc-r2` (`verdict: pass`, ratio
  0.5453 — 14.59 vs 26.76 ms) — five retained AB/BA rounds each with five
  matched custom/Optax comparisons, the batch GPU clocks/power telemetry
  CSV bundled and hashed inside each receipt, UUID-bound identity. The
  2.0x gate passes by receipt-recomputed qualification (the initial
  publications recorded `verdict: fail` with
  `performance-qualification-not-implemented` because the validator had no
  performance kind; those receipt dirs were retired at `cdf62708a` when
  the implemented qualifier attested the pass). Custom is faster than
  Optax in both regimes.
- Phase 8: solver-matrix reconciliation after gates (`a8134ebc8`);
  independent adversarial review by the plan author (no P0s; all three P1s
  fixed: `e3e8d5fe9`, `b243a176e`, `03de61652`); ordered commits with
  detached-worktree replays at three successive SHAs.

Negative result (do not retry blindly): bitwise-safe batching of the
matupd shift loops plus vmap of the two-loop fixed curvatures lowered the
fused StableHLO while-count 18→14 but raised dynamic_slice 53→59 and showed
no warm-time improvement in interleaved A/B on the RTX 5090 (clean pairs
1.03x/1.40x vs baseline); reverted per the Phase-6 revert rule. Matvec
batching of the update-row dots diverges from the sequential loop at 1 ULP
(deterministic n=2 case) and is unusable in shared transition code. The
equivalence-test artifact is preserved in the session scratchpad.

Deferred-items sweep (2026-08-03, second pass — commits `4127102e7`,
`af4a7619f`):

- Warm soak CLOSED: `benchmarks/lbfgs_warm_soak.py` (same-process, prepared
  once, fail-closed plateau verdict). Definitive clean-worktree 20-run RTX
  5090 soak: plateau on all axes — executable count 1 throughout, RSS drift
  20 KiB (slack 2048), VRAM flat at 968 MiB, warm median ratio 1.083
  (limit 1.2). Artifact tracked at
  `docs/receipts/custom-quasi-newton/warm-soak-gpu-af4a7619f/` and
  mirrored in the same-host archive (generated at
  `.artifacts/custom-quasi-newton/final-warm-soak-gpu-af4a7619f.json`).
- Exact-inner replay CLOSED — and it caught a real regression: the
  accepted-incumbent controller postdates the exact-inner receipt and its
  host boundary used implicit H2D transfers, crashing guarded runs
  (`JAX_TRANSFER_GUARD=disallow`). Fixed via explicit `runtime_device_put`
  (`4127102e7`, with two transfer-guard regression tests). Guarded CPU and
  RTX 5090 replays now complete with inner residuals at receipt scale
  (2.18e-29 / 6.60e-15 CPU; 2.28e-29 / 6.76e-15 GPU vs the receipt's
  1.98e-29 / 6.30e-15 and 2.52e-29 / 7.09e-15).
- StableHLO pair CLOSED at
  `docs/receipts/custom-quasi-newton/compile-shape-fused-v2-cd6b66368/`
  (tracked, archive-mirrored): custom records
  `solver_route=fused_stepwise`, `git_clean=true`, an available candidate
  SHA, and exactly three compiled executables; Optax records
  `optax_lbfgs` with two. The first pair
  (`final-compile-shape/`, retired) was INVALID — the script did not
  thread `--intent` into preparation and measured the stepwise route while
  labeled fast; fixed in the review iteration (route-aware program
  enumeration) and regenerated from a pristine worktree at `cd6b66368`.
- Pyright/compileall CLOSED with adjudication: compileall green; the repo's
  scoped Pyright gate has 50 pre-existing errors, all in files this
  campaign never touched (the one touched gated file,
  `parity_tolerances.py`, is clean); differential on the non-gated campaign
  files is 61 → 68 (+7 latent annotations in heavily-edited files),
  recorded as residual debt. Caution: Pyright silently excludes
  dot-directories — a baseline run against a worktree under `.artifacts/`
  analyzes nothing and reports a vacuous 0.
- Detached-checkout archive replay CLOSED (validate-all --archive-root over
  the complete 47-receipt mirror).

Different-host archive replay CLOSED 2026-08-03: landau returned to
service (user-confirmed) and the full authority bundle replayed green
there — see `offhost-replay-landau-20260803.md` and the ticked Phase-7
box above. Still deferred: optional line-search floor-acceptance fix for
the stochastic GPU deaths (unauthorized).

## External review iterations (2026-08-03 addendum)

Two external adversarial Crucible verdicts (both FAIL_ITERATE) were worked
finding-by-finding, each finding proven against the live repo before any
edit. Fix commits, in order:

- `0e686d682` — CRITICAL: the endpoint certificate trusted
  `provider_success` before interpreting statuses; termination evidence is
  now normalized per provider convention and contradictions fail closed.
  The runtime's duplicate classifier was deleted (single owner).
- `80bf41ea1` — compile-shape `--intent` threading; removal of the unused
  fused result-payload compilation; warm-soak monitor `finish()` in
  `finally`; RUF059 cleanups.
- `72b63cdae` — example finalization consumes one evaluation anchored at
  the accepted incumbent's inner state (baseline-owned finalization calls
  removed), with structural tests proving anchor identity.
- `cd6b66368` — route-aware compile-shape program enumeration (the fused
  route was unmeasurable: preparation tripped the executable-reuse guard);
  live regression locks fused preparation at exactly three executables.
- `45f853a12` — dedicated `optax-lbfgs` status convention (Optax status 2
  is a line-search failure, not an evaluation limit); cross-convention
  contamination locked; anchored-forward structural expectations repaired.
- `e861f9355` — receipt provenance authentication: baseline gradients must
  bind to record 0's parameter hash; runner `git_commit` must be a
  canonical 40-hex SHA resolving to a commit object at publish and
  validate time; child `gpu_memory`/`trial_trace` artifacts bound by
  containment and recomputed SHA-256; performance-kind qualification
  implemented (matched pairs, route enforcement, warm-median ratio <= 2.0
  gate); publication snapshots before validation.
- `cdf62708a` — performance receipts republished with attested pass
  verdicts (`-r2` IDs); telemetry bundled; compile-shape v2 and warm-soak
  artifacts tracked and archive-mirrored.

Third external pass (delta review, same day) returned FAIL_ITERATE with
three P1s, fixed in the next commit batch:

- Status conventions were re-derived from the ACTUAL emitters, one table
  per solver implementation (`scipy-bfgs`, `private-bfgs`, `host-bfgs`,
  `scipy-lbfgsb`, `private-lbfgsb`, `host-lbfgsb`, `optax-lbfgs`): the
  custom benchmark L-BFGS lane runs the private SciPy port whose status 2
  is a line-search failure (not `host-lbfgsb`'s evaluation-limit), and the
  private BFGS reverses SciPy's 2/3 (2 = nonfinite backstop, 3 =
  line-search). `status_convention_for` now requires the
  accepted-incumbent discriminator (custom BFGS: host core under
  continuation vs private on-device solver), the example names its host
  emitters directly, merged budget statuses (`private/scipy-lbfgsb` 1) are
  discriminated into iteration- vs evaluation-limit by the iteration
  evidence, and real-emitter failure tests drive the actual solvers
  (SciPy BFGS, private BFGS, private L-BFGS-B, host BFGS) into each
  reachable failure path — the private BFGS nonfinite backstop is
  unreachable through the hardened line search and is locked as the
  documented sentinel behavior instead. All 47 receipts revalidate
  unchanged under the emitter-true tables.
- Receipt source-run aliasing closed: `validate-all` now requires each
  source-run name to be one canonical filesystem component, so
  path-normalized aliases (`./round-1`, `round-1/`) can no longer
  multiply one physical run's samples into the performance medians.
- Phase-8 evidence made true rather than claimed: routing inventory
  gained the `fused_stepwise` addendum, the solver matrix gained the
  migration/cache/rollback record and receipt-bound medians, and the
  rollback/replay tick now states exactly what its receipts prove.

Fourth external pass (delta review, same day) returned FAIL_ITERATE with
two P1s and two P2s, fixed in the next commit batch:

- Emitter identity made durable in the persisted bytes: the runtime now
  splits the custom BFGS route by emitter
  (`custom_bfgs_host_incumbent` / `custom_bfgs_private`), and receipt
  recomputation derives the status convention purely from each row's
  recorded `solver_route` (with a frozen transcription pinning the
  pre-split `custom_bfgs_stepwise`+`boozer` rows to the host core) —
  the live fixture registry is no longer consulted at validation, so
  later registry policy changes cannot invalidate authentic history
  (regression: validation passes with the registry poisoned). Unmapped
  route/case combinations fail closed.
- The fused-route rollback lever was actually rehearsed
  (`rollback-rehearsal-fused-20260803.md`): one-line `_solver_route`
  rollback in a clean worktree at `76f8fdf23` — fast intent falls back
  to stepwise, SciPy trajectory parity stays green, exactly the two
  fused route pins fail, and the receipt gate caveat is recorded. The
  matrix and Phase-8 prose now cite it (the 2026-08-02 rehearsal covers
  only the pre-fused prepared-runtime commits).
- Off-host replay wording now distinguishes the validator checkout
  (`487d9ff89`) from the receipts' candidate SHAs and states that no
  solver execution was replayed.
- A real SciPy-BFGS failure test (NaN -> status 3 -> nonfinite;
  maxiter -> 1 -> iteration-limit) closes the emitter-coverage gap, and
  the coverage claim is scoped to reachable paths.

Reviewer sub-claims refuted with evidence (not re-fixed): the batch
telemetry CSVs existed at
`.artifacts/custom-quasi-newton/final-{359fd41fc,7b1372ad0}-gpu-telemetry.csv`
(the `359fd41fc` CSV — the one belonging to the receipts' lane batch — is
now bundled and hashed inside each `-r2` receipt; the `7b1372ad0` CSV
belongs to the superseded earlier batch and remains host-local plus
archive-mirrored); 17 of the 18 failing tests in
`tests/geo/test_surface_objectives_jax.py` fail identically at the commit
that introduced them (other-workstream drift, proven in a detached
worktree) — the eighteenth, the general-only structural test, was a real
campaign regression and is fixed in `45f853a12`.

## Open Questions

- Does the first rejected Boozer trial fail in the predictor, primal Newton
  solve, adjoint solve, or Wolfe evaluation?
- Can objective-owned bounded recovery close the candidate domain without an
  optimizer-level protocol?
- Does matched profiling confirm that `NEW_X` reentry dispatch is the useful
  L-BFGS fusion boundary?
- Does `fused_stepwise` compile within the watchdog on RTX 5090 and A100?
- Which durable off-host location will own the promotion bundle?
