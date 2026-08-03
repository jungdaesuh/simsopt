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
semantic validator. Tests must cover the current version, the allowed legacy
versions, unknown-version rejection, missing/null/wrong-type fields, nonfinite
metrics, tampered derivations, and historical round trips.

## Implementation Plan

1. Freeze failure, finalization, and schema behavior.
   - [ ] Add a RED example regression proving a finite, decreasing Boozer result
         with `outer_solver_success=false` must produce top-level
         `status="failed"`.
   - [ ] Add zero-step cases proving stationary raw-success is valid and the
         current nonstationary Boozer zero-step result is invalid.
   - [ ] Add a RED runner/receipt test requiring the trial fields needed to
         diagnose line-search failure.
   - [ ] Preserve the exact-inner CPU/GPU test as a GREEN non-regression gate.
   - [ ] Implement the version 7/version 2 schema matrix above before producing
         new promotion receipts.
   - [ ] Make receipt validation recompute outer and scientific success from
         raw fields; never trust top-level `status` or a stored aggregate alone.

2. Diagnose the Boozer outer failure with one bounded owner.
   - [ ] Add a benchmark-only host diagnostic harness with immutable trial
         records. Correlate optimizer and objective data by
         `(evaluation_index, SHA-256(float64 parameter bytes))`; use no global
         mutable state.
   - [ ] The host line search owns trial step length and Wolfe errors. The
         solved-pair/objective boundary owns predictor, primal, adjoint, Newton,
         residual, physical objective, and gradient data. The diagnostic
         harness is the only join owner.
   - [ ] Record predictor kind/success, primal and adjoint success, Newton
         status/iterations, inner residual, raw and filtered objective,
         gradient finiteness/norm, step length, and Wolfe errors.
   - [ ] Keep capture optional and bounded. Normal execution retains no trial
         trajectory and adds no host callback.
   - [ ] Run matched native/custom CPU diagnostics, then the custom strict RTX
         5090 diagnostic, and identify the first causal divergence.
   - [ ] Record the selected recovery design and rejected alternative in this
         plan before changing objective or optimizer behavior.

3. Fix Boozer finalization and the diagnosed root cause with RED -> GREEN ->
   REFACTOR.
   - [ ] Make the example finalizer require certified outer and inner success,
         finite certificate fields, and the Scientific Certificate above.
   - [ ] Write an observable RED regression for the selected candidate failure;
         preserve the failing pre-fix command, output, commit, and hash.
   - [ ] If objective-owned recovery is selected, implement one immutable,
         fixed-length policy in `surface_objectives_traceable.py` and report its
         attempts/evaluations.
   - [ ] If the optimizer protocol is selected, define one typed private
         outcome and inventory every BFGS/L-BFGS caller before changing it.
   - [ ] Require every accepted trial to use a certified inner solve and finite
         physical objective/gradient; never certify a penalty value.
   - [ ] Refactor only after GREEN, keeping candidate classification and retry
         policy under one owner.

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
   - [ ] Emit fixture-build, provider-preparation, first-execution, and warm
         times as separate synchronized fields. Keep cold total derived.
   - [ ] Give custom and Optax the same work-counter, transfer, RSS, and
         process-attributed VRAM boundaries. Mark genuinely unavailable values
         unavailable rather than inferring them.
   - [ ] Extend `benchmarks/lbfgs_ondevice_compile_shape.py` to select custom or
         Optax and compile the exact provider factories used by the runner.
         Record route, per-program and aggregate StableHLO bytes, compile time,
         executable count, dtype, fixture, and options. Share benchmark
         preparation code; do not copy optimizer equations.
   - [ ] Profile objective/line-search work, per-kernel device time, and
         host-transfer latency. Confirm or reject the reentry/synchronization
         hypothesis before Phase 6.
   - [ ] Add version-7 boundary tests proving fixture construction and compile
         diagnostics are excluded from warm measurements.

6. Evaluate the explicitly routed L-BFGS fast design.
   - [ ] Add a RED routing test: `fused_stepwise` must reach the new prepared
         program, while `stepwise` and `monolithic_debug` retain their current
         paths and cache identities.
   - [ ] Add a RED transfer test against the measured baseline of two `advance`
         observations per accepted iteration. The fast target is no more than
         `iterations + 1` advance observations, with exact leaves and bytes
         retained in the receipt.
   - [ ] Implement the fused reentry-plus-search program only after Phase 5
         confirms that target is causal.
   - [ ] Preserve status, counters, callbacks, zero budgets, `maxfun`, bounds,
         inverse-Hessian behavior, and accepted-state scientific parity.
   - [ ] Abort if compilation exceeds 120 seconds or 8 GiB RSS, duplicates
         transition mathematics, changes the endpoint contract, or recreates
         the rejected all-entry generic branch graph.
   - [ ] A performance receipt is eligible only when `intent=fast` and
         `solver_route=fused_stepwise`; the validator rejects any other route.
         Current `stepwise` timing cannot satisfy this gate.
   - [ ] Keep the route only if warm custom time is no more than `2.0x` matched
         Optax on both GPUs and improves over the current custom median. Revert
         a pure performance regression.

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
   - [ ] Repeat the matched `coil47` fast comparison on RTX 5090 and A100 with
         FP64 and no CPU fallback.
   - [ ] Bind every StableHLO and compile artifact to its provider, solver
         route, candidate SHA, and exact device UUID; require a complete
         custom/Optax artifact pair for each GPU.
   - [ ] Publish version-2 manifests and raw artifacts from a clean detached
         candidate checkout.
   - [ ] Add `validate-all --archive-root` so a mounted immutable copy can be
         verified without rewriting receipt manifests. Map by receipt ID,
         verify bundle inventory hash/storage identity, and reject incomplete
         remaps.
   - [ ] Copy the authority bundle to a different host or durable store, mount
         it read-only, and replay from a detached checkout at the recorded
         candidate SHA with the recorded environment lock.

8. Complete rollout and API review.
   - [ ] Inventory observable behavior changes and every caller of
         `lbfgs_run_mode`, Boozer example finalization, runner schema, and
         receipt schema.
   - [ ] Document the new `fused_stepwise` option, its explicit fast callers,
         parity preservation, migration path, cache behavior, and rollback.
   - [ ] Update the solver matrix and provider plan only after gates pass.
   - [ ] Record ordered implementation commits and rehearse rollback in a clean
         worktree.
   - [ ] Obtain independent architecture, numerical, runtime, and evidence
         review with no unresolved finding.

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
- [ ] Authority artifacts validate from a durable off-host copy.

## Open Questions

- Does the first rejected Boozer trial fail in the predictor, primal Newton
  solve, adjoint solve, or Wolfe evaluation?
- Can objective-owned bounded recovery close the candidate domain without an
  optimizer-level protocol?
- Does matched profiling confirm that `NEW_X` reentry dispatch is the useful
  L-BFGS fusion boundary?
- Does `fused_stepwise` compile within the watchdog on RTX 5090 and A100?
- Which durable off-host location will own the promotion bundle?
