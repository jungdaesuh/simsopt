# Single-Stage JAX GPU Native-Equivalent No-Hit Diagnostic Plan

**Status:** In progress  
**Last updated:** 2026-08-11

## Purpose

This file is the single source of truth for `NEQ-GNTR1-DIAG1`, a prospective,
diagnostic-only replay of the unchanged `NEQ-GNTR1` numerical route. It must
identify why the route did not trigger native-equivalent quality within its
frozen bound, while adding no optimizer decision, tuning parameter, endpoint
replacement, or promotion path.

The numerical route remains `NEQ-GNTR1`. `NEQ-GNTR1-DIAG1` names only the
evidence producer, schema, profiler annotations, and fail-closed adjudication.

## Goals

- Retain every fixed-shape attempt-history row and independently recompute its
  active prefix, outcome histogram, counters, and certificate semantics.
- Retain the unconditional post-timing terminal state, objective, all 255 raw
  and scaled equalities, KKT telemetry, accepted-ledger binding, and signed
  quality margins even when the device quality latch is false.
- Attribute aggregate GPU work to frozen current-model, Steihaug, trial, and
  correction phase families without callbacks, hot transfers, or internal
  synchronizations.
- After one passing production-shape lower/compile-only preflight, execute
  exactly one isolated RTX 5090 cold diagnostic and select exactly one next
  development route from valid raw evidence.

## Non-Goals

- Changing the GN model, HVP, projection, correction, acceptance, trust-radius
  update, termination order, quality predicate, tolerances, scaling, or branch.
- Producing a warm timing result, native speed win/loss, formal comparison,
  endpoint promotion, A100 portability result, or scientific convergence claim.
- Recovering row-level evidence from the sealed `0252Z` artifact. Those bytes
  never retained the history or no-hit endpoint and cannot be rewritten.
- Timing individual attempts by inserting clocks, host callbacks, transfers,
  synchronizations, or profiler start/stop operations inside the device loop.

## Current Context

- Live repository HEAD before this plan: `52dea17ddf3012cf923fc92da78c0d73a17f4625`.
- Closed numerical core SHA-256 before diagnostic work:
  `d6e36e88d9cb8998ea033d20f034169d7c263bea56479bddf00983cf3eda6159`.
- Closed NEQ adapter SHA-256 before diagnostic work:
  `0c0875cf068d0e2ee5e1ca46ed49deb085ebaa944ebef825ba66730717151d4b`.
- Closed runner SHA-256 before diagnostic work:
  `7d37b74a539363f82c2099b773de41c0ecf1cf94e6b4a3abb20470bbbafbf8d3`.
- Native reference manifest SHA-256:
  `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.
- Sealed cold artifact manifest SHA-256:
  `1f4a4b99bc22d2080619e43366be1f78b5a9b97efd2319d3b90ead1cfbe7ad6e`.
- Its producer SHA-256 is
  `16e106212655073a84c6d1ca7d660ec684a34eb3610783291f4110ee474f6df6`.
  It retained 300 attempts, 203 accepts, no quality latch, and a synchronized
  solve time of `296.461143728 s`, but not the attempt rows or no-hit endpoint.

The live worktree is intentionally dirty and contains untracked route files.
All source snapshots and manifests must bind the exact executed bytes; no claim
may rely on HEAD alone.

## Rationale

The existing optimizer already owns a fixed `maximum_attempts` history with
outcomes, objectives, feasibility, reductions, radii, step norms, correction
certificates, Steihaug telemetry, and curvature/projection evidence. Its split
finalizer already recomputes the returned endpoint and KKT evidence after the
timed loop. The prior producer discarded these values when the quality latch
was false. The smallest diagnostic therefore serializes existing evidence and
adds route-owned margins after timing.

Two designs were considered:

1. Add route-specific quality fields or callbacks to the generic hot loop.
   This was rejected because it changes the loop pytree, writes extra device
   state, leaks native policy into the generic optimizer, and weakens the
   unchanged-path claim.
2. Reuse the exact compiled loop, add metadata-only opt-in named scopes, and
   construct a new post-timing diagnostic result and receipt. This is selected.
   Threshold knowledge remains in the NEQ adapter/receipt, while the optimizer
   retains only generic scalar evidence.

The old and new full GPU trajectories cannot be compared row for row because
the old artifact omitted the rows. Unchanged behavior must instead be proved by
source/control-flow audit, exact CPU/JIT pytree equivalence with scopes disabled
and enabled, unchanged policy hashes, and prospective aggregate comparison to
the retained `300/203/no-hit` facts. Aggregate divergence is evidence, not an
automatic numerical invalidity.

## Assumptions

- "All 300 attempt rows" means the existing fixed scalar history of 300 slots.
  The validator requires an active prefix of length `attempts` and inactive
  padding thereafter. A fatal exit, quality hit, or 256 accepts may truthfully
  produce fewer than 300 active rows without changing termination semantics.
- Phase timings are aggregate profiler attribution, not exact per-attempt wall
  times. A missing or ambiguous trace yields `PHASE_TIMING_NOT_PRODUCED` without
  invalidating otherwise complete numerical diagnostics.
- The profiled cold solve time is diagnostic-only because no unprofiled paired
  control is authorized.

## Frozen Numerical Contract

- Problem dimensions: 716 optimizer/physical state components, 255 exact
  equalities, and 2110 objective-residual components.
- Physics and objective: identical frozen input bundle, fixed/free mapping,
  objective ledger and weights, Boozer residual ordering/orientation, volume
  equality, and continuation-connected branch contract.
- Precision: JAX x64 enabled and all authoritative state, residual, gradient,
  Jacobian, multiplier, and equality evidence in FP64.
- Native objective target: `4.4822246533126125e-08`.
- Scaled feasibility tolerance: `1e-10`.
- Componentwise raw equality bound:

  ```text
  abs(q_gpu[i]) <= abs(q_native[i]) + 1e-12 + 1e-10*abs(q_native[i])
  ```

- The exact `NativeEquivalentQualityPolicy.policy_sha256`, native equality
  vector/hash, bootstrap anchor, variable scale, constraint inverse scale, and
  reference artifact are source-manifest inputs.
- `NEQ_GNTR1_OPTIONS` remains the existing CFS policy with only
  `maximum_accepted_steps=256` and `maximum_attempts=300` replaced. No option or
  default may change.
- Termination precedence remains fatal evidence, device quality latch, 256
  accepts, then 300 attempts. No diagnostic work participates in this order.

## Diagnostic Evidence Contract

### Attempt history

Retain all fixed 300 slots for every existing history field, including:

- outcome, accepted-step number, current/candidate objective and feasibility,
  and current stationarity;
- actual/predicted reduction, reduction ratio, current/next radius;
- tangent, correction, and applied norms and correction/radius ratios;
- Steihaug iterations, HVP evaluations, termination, boundary flag, projected
  residual target/final norm, and residual-projection certificates;
- terminal/probe curvature, direction rotation, residual value/gradient
  defects, HVP symmetry, correction residual/forward bound, trial Gram
  factorization/solve residuals, and current-projection certificates.

The independent validator derives counts directly from `history.outcome`:

- `attempts = count(outcome != INACTIVE)`;
- `accepted_steps = count(outcome == ACCEPTED)`;
- `retryable_rejections = count(outcome in RETRY_*)`;
- the active rows form an exact prefix and every remaining row is `INACTIVE`;
- the recorded terminal counters/status/fatal/bounded flags agree with those
  rows and the frozen termination policy.

Nonapplicable floating evidence is encoded as JSON `null` with the applicable
status/outcome retained; NaN and infinity are forbidden in canonical JSON.

### Unconditional terminal endpoint

After the timed loop is synchronized, the existing finalizer must run once even
when no latch fires. Retain raw FP64 arrays plus shape, dtype, and little-endian
content hash for:

- optimizer coordinates and reconstructed physical state;
- all 255 raw and scaled equalities;
- objective gradient, multipliers, and raw stationarity;
- the accepted optimizer/physical ledger and mask.

Retain the authoritative objective and five raw objective terms/weights,
scaled feasibility, raw physical KKT infinity norm, scaled stationarity,
typed KKT availability, residual reconstruction/transpose evidence, final
projection/multiplier certificates, and all finite/status fields.

The last valid accepted-ledger row must equal the returned final coordinates,
and the valid ledger count must equal `accepted_steps + 1` including bootstrap.
The finalizer may not project, replace, restore, or otherwise change that state.

### Signed quality margins

The receipt recomputes the following from raw values. Positive is pass, zero is
the exact boundary, and negative is fail:

```text
objective_margin = 4.4822246533126125e-08 - objective
component_margin[i] = abs(q_native[i]) + 1e-12
                      + 1e-10*abs(q_native[i]) - abs(q_final[i])
scaled_feasibility_margin = 1e-10 - max(abs(D*q_final))
```

Retain all 255 component margins, their minimum and lowest index, plus objective,
component, and feasibility usage ratios. Indices are zero-based; ties use the
lowest index. The ratios are exactly
`objective/4.4822246533126125e-08`,
`max_i(abs(q_final[i])/(abs(q_native[i])+1e-12+1e-10*abs(q_native[i])))`,
and `max(abs(D*q_final))/1e-10`. Residual value, residual gradient, and
transpose margins use their frozen `1e-12`, `1e-10`, and `1e-10` thresholds.
KKT remains non-gating telemetry.

The existing NEQ finalizer does not expose raw physical multipliers,
stationarity, or KKT. The diagnostic successor must perform exactly one
additional post-timing endpoint diagnostic at the unchanged returned
coordinates, using the final scaled multipliers and the canonical
`cfs_sqp1_endpoint_diagnostics` authority. This device computation is a
separately timed `terminal_endpoint_diagnostics` phase after the loop and before
the one final D2H. Its state hash must equal the loop's last valid accepted row.

### Aggregate GPU phase attribution

Only an opt-in diagnostic trace context may activate these stable named scopes:

- `gntr.current_linearization`
- `gntr.current_certificates`
- `gntr.steihaug`
- `gntr.trial_evaluation`
- `gntr.nonlinear_correction`
- `gntr.corrected_candidate_evaluation`
- `gntr.acceptance_radius_update`

The disabled production path emits no scope wrapper into the numerical pytree.
The diagnostic trace may annotate existing operations but may not add a value,
branch, callback, transfer, synchronization, or computation to the loop.

The artifact retains the raw profiler trace and independently assigns each
device interval to its deepest recognized scope. An interval with equal-depth
recognized owners is ambiguous and invalid. The per-phase duration is the union
of intervals assigned to that phase; the total attributed duration is the union
across all seven recognized phases after deepest-owner assignment. Intervals
owned by different phases may overlap in time on different GPU streams, so the
total union is less than or equal to the sum of per-phase unions. Retain every
pairwise/inter-phase overlap and unattributed device time separately. A complete
nonfatal cold requires all seven phases to be present, a finite positive total
device-active union, and
`total_attributed_seconds/total_device_active_seconds >= 0.90`. It must never
use host `perf_counter` intervals around asynchronous
internal work as device phase timings. Unsupported scope identity, ambiguous
attribution, or invalid interval unions produces `PHASE_TIMING_NOT_PRODUCED`,
which makes the overall diagnostic incomplete and ends this one-run protocol.

## Timing and Execution Contract

One new absent output root outside the repository owns one immutable source
snapshot and one copied validated native reference. The integrated schedule is:

1. exactly one fresh-process production-shape lower/compile-only preflight;
2. only if its independently recomputed gate passes, exactly one new pristine
   RTX 5090 diagnostic `cold` child from the same snapshot;
3. no other child, sample, replay, replacement, or warm run.

The parent must reject a nonmatching RTX UUID before output creation. Both
children use the exact requested virtual-environment interpreter, isolated
snapshot cwd, `JAX_PLATFORMS=cuda`, x64, one visible GPU, disabled persistent
compilation cache, and `XLA_PYTHON_CLIENT_PREALLOCATE=true`.

Preflight activates the identical diagnostic annotation context and phase-schema
hash used by the cold child, then lowers/compiles that exact annotation-enabled
loop/finalizer/map and verifies zero callbacks. It does not start a profiler or
dispatch the executable. It
records `solver_dispatched=false`, `finalizer_called=false`,
`endpoint_audit_called=false`, and `campaign_authorized=false`. Failure seals a
terminal artifact and prohibits the cold child.

After compile and state readiness, the cold child starts the profiler trace
immediately before the one loop dispatch. Cold timing has the same start and
ends only after `jax.block_until_ready(loop_result)`. The child then stops the
profiler before any finalizer, endpoint diagnostic, D2H, or export. Only after
the stop completes may it run the post-timing finalizer and raw endpoint
diagnostic, perform the one final D2H, and export/serialize the trace. The
transfer guard is `disallow`. Compile, state preparation, finalizer, endpoint
diagnostic, final D2H, profiler export, serialization, and validation are
outside the solve interval. The producer retains monotonic timestamps for
profiler start/stop, compile, state-ready, solve, finalizer, endpoint diagnostic,
D2H, trace export, serialization, and total process boundaries, and rejects
trace device events outside the loop envelope. The profiled solve duration is
not compared with native time.

Parent evidence binds exact PID, start ticks, full procfs argv, interpreter,
GPU UUID, raw memory samples, physical device bytes, peak bytes/fraction, source
identity before/after each child, bounded stdout/stderr plus full hashes/sizes,
runtime/import/environment evidence, and API-level transfer/callback evidence.

## Verdict and Next-Route Contract

The outer numerical verdict is derived by this exclusive priority table:

1. `DIAGNOSTIC_INCOMPLETE` if the preflight gate fails; the cold process is
   missing, times out, crashes, reports fatal/protocol/monitor/resource/source
   failure; any required raw numerical/phase evidence is absent/nonfinite; or
   phase attribution is `PHASE_TIMING_NOT_PRODUCED`.
2. Otherwise `DIAGNOSTIC_COMPLETE_QUALITY_HIT_NONPROMOTING` when the device
   quality latch is true and its first-hit counters and raw predicate inputs
   recompute true.
3. Otherwise `DIAGNOSTIC_COMPLETE_NO_HIT` when the latch is false and its
   first-hit counters are zero.

Historical aggregate relation is a separate orthogonal field, never an outer
verdict: it is `MATCHES_RETAINED_AGGREGATES` exactly for 300 attempts, 203
accepts, and latch false; otherwise `DIVERGES_FROM_RETAINED_AGGREGATES`. A
quality hit may therefore be complete while truthfully diverging from the old
aggregate.

Every outcome is nonpromoting. `engineering_campaign_receipt_produced=false`,
`promotion_authorized=false`, and formal comparison `NOT_PRODUCED` are literals.
The new artifact must not contain a warm sample or an engineering speed verdict.

For complete, finite, source-valid evidence, select exactly one next route in
this fixed priority order:

1. `RETRY_MODEL_REUSE` when

   ```text
   attributed_current_model_seconds = duration(union(
       intervals owned by gntr.current_linearization or
       gntr.current_certificates))
   reuse_opportunity_estimate =
       retryable_rejections / attempts
       * attributed_current_model_seconds / total_attributed_seconds
   ```

   is finite and at least `0.05`. This rule additionally requires the frozen
   `0.90` attribution-coverage gate, all seven required phases, a finite positive
   denominator, and validator proof that
   `0 <= attributed_current_model_seconds <= total_attributed_seconds`.
2. Otherwise `RADIUS_RETRACTION` when `min(component_margin)` or the scaled
   feasibility margin is negative, or when correction-certificate,
   feasibility, and step-bound retries together are at least `0.10*attempts`.
3. Otherwise `CONDITIONING_MODEL_CHANGE`.

The reuse quantity is explicitly an attribution-based planning estimate, not a
measured speedup. `0.05` and `0.10` are diagnostic development-selection
thresholds, not optimizer parameters. If phase timing is not produced, rule 1
cannot pass. Corrupt, incomplete, nonfinite, or authority-invalid evidence has
no scientific route selection and remains `DIAGNOSTIC_INCOMPLETE`.

## Artifact and Receipt Contract

- Use a new strict canonical schema rooted at
  `single-stage-neq-gntr1-no-hit-diagnostic-v1`; never mutate the old receipt.
- The validator accepts exact keys and types only, rejects bool-as-number,
  duplicate JSON keys, noncanonical encoding, NaN/infinity, path traversal,
  symlinks, writable paths, missing/extra files, and digest/size/mode drift.
- Raw arrays and trace bytes are content-addressed artifact references, not
  trusted producer summaries.
- The exact-set manifest includes plan, source snapshot, runner, receipt,
  validator, tests, reference, producer, terminal, raw arrays, trace, runtime,
  process, and memory evidence. Files are `0444` and directories `0555` only
  after independent validation succeeds.
- The validator recomputes every counter, margin, phase union, resource gate,
  numerical verdict, and next-route selection. Producer booleans never decide.
- Preserve the immutable `0252Z` artifact by reference/hash only; do not copy it
  into a role that implies complete campaign authority.

## Implementation Plan

1. Freeze this SSOT and route identities.
   - [ ] Independent math and provenance reviews approve the formulas,
         nonperturbation boundary, schedule, schemas, and selection policy.
   - [ ] Record the final SSOT SHA-256 before any production preflight.
2. Add diagnostic evidence without changing numerical decisions.
   - [ ] Add opt-in named phase scopes and exact disabled/enabled pytree parity
         tests.
   - [ ] Prove enabled and disabled production-shape jaxpr/StableHLO have the
         same numerical operations, constants, shapes, dependencies, and
         control flow after stripping only approved name/location metadata.
   - [ ] Add an additive NEQ diagnostic result exposing unconditional endpoint
         evidence and signed margins, plus one state-bound raw physical KKT
         diagnostic, while reusing the exact compiled loop.
   - [ ] Preserve existing public result arities and callers; inventory and test
         every affected API.
3. Add the independent diagnostic receipt and validator.
   - [ ] Serialize all fixed history slots, terminal arrays, ledger, raw
         profiler trace, runtime/process/memory evidence, and timestamps.
   - [ ] Recompute all raw semantics, verdicts, and route selection with a
         comprehensive mutation matrix.
4. Add the integrated one-preflight/one-cold runner mode.
   - [ ] Make diagnostic mode mutually exclusive with campaign, preflight-only,
         and internal child modes.
   - [ ] Prove the call schedule is exactly `[preflight, cold]`, or only
         `[preflight]` on gate failure, and cannot create warm directories or
         `campaign.json`.
   - [ ] Seal truthful compile/OOM/timeout/crash/protocol/monitor/source-drift
         terminal evidence.
5. Qualify and execute.
   - [ ] Run focused and integrated CPU tests, Ruff check/format, compileall,
         type/static checks where configured, diff checks, source/hashes audit,
         and fresh independent review.
   - [ ] Run exactly one production-shape lower/compile-only preflight on the
         frozen snapshot and independently validate it.
   - [ ] If and only if it passes, run exactly one cold diagnostic child and no
         other numerical process.

## Validation Plan

- [ ] Existing core/adapter/runner/receipt tests remain green before and after.
- [ ] Exact leaf-by-leaf equality holds between the old loop and diagnostic
      loop for deterministic CPU/JIT fixtures, including all history fields,
      coordinates, counters, statuses, ledger, mask, and latch metadata.
- [ ] Existing `NEQ_GNTR1_OPTIONS`, policy payload/hash, predicate thresholds,
      control-flow decisions, and termination order are byte- or AST-audited
      unchanged.
- [ ] Named scopes add only source metadata and the compiled executable contains
      no Python/host callback.
- [ ] Annotation-enabled preflight and cold lower the same phase-schema identity;
      normalized StableHLO equivalence proves scopes do not alter numerical
      dataflow relative to the disabled route.
- [ ] Margin tests cover exact threshold, one ULP above/below, worst component,
      nonfinite evidence, and KKT non-gating behavior.
- [ ] History tests cover full 300-row no-hit, early fatal, early quality hit,
      256-accepted completion, inactive padding, every outcome, and counter
      recomputation.
- [ ] Receipt mutations cover every raw scalar/vector/hash, timestamp ordering,
      phase overlap/unattributed handling, memory capacity/fraction, process
      identity, source drift, selection priority, and manifest closure.
- [ ] A real snapshot smoke proves the exact entrypoint/import/runtime closure.
- [ ] Preflight artifact proves compile-only behavior, zero callbacks, correct
      GPU/FP64/dimensions/policy, memory below `0.8`, and no cold authorization
      unless its independently derived gate passes.
- [ ] Cold artifact proves exactly one loop dispatch, no warm/tuning/promotion,
      zero guarded hot transfers/callbacks, one final post-timing D2H, and a
      complete independently recomputed diagnostic verdict.

## Risks and Mitigations

- Risk: named scopes or trace collection perturb compilation or runtime.
  Mitigation: exact numerical pytree parity; report profiled timing only as
  diagnostic; make phase-timing failure separable.
- Risk: schema code duplicates NEQ quality policy.
  Mitigation: adapter owns margin construction; receipt consumes raw policy
  values and independently recomputes formulas from one frozen contract.
- Risk: unconditional endpoint publication is mistaken for endpoint audit.
  Mitigation: label it terminal diagnostic evidence; prohibit branch/cross-
  evaluator promotion and retain `promotion_authorized=false`.
- Risk: the new trajectory differs from the sealed aggregate.
  Mitigation: retain both; set the orthogonal aggregate-relation field to
  `DIVERGES_FROM_RETAINED_AGGREGATES` and do not rewrite, discard, replace, or
  tune after seeing the result.
- Risk: the dirty worktree obscures execution identity.
  Mitigation: immutable exact-byte snapshot with tracked/untracked manifests,
  pre/post rehash, runtime/import evidence, and exact-set artifact manifest.

## Rollback Plan

Before production execution, rollback is deletion of the new diagnostic-only
files and removal of opt-in scope calls; the existing route and artifacts remain
untouched. After a diagnostic artifact is published, never delete or mutate it.
Any successor must reference its hashes and use a new SSOT/schema identity.

## Completion Criteria

- [ ] The final SSOT, changed source files, and exact validation commands and
      complete green outputs are pasted with hashes.
- [ ] Independent reviews prove the frozen physics, objective, dimensions,
      equalities, FP64/scaling/branch policy, quality predicate, and optimizer
      decisions remain unchanged.
- [ ] Exactly one passing lower-only preflight and at most one authorized cold
      diagnostic are sealed under new absent output roots; no warm or tuning run
      exists.
- [ ] The canonical artifact independently reproduces the full outcome
      histogram, terminal quality margins/KKT, mandatory phase
      attribution/resource evidence, verdict, historical aggregate relation,
      and exactly one valid next-route selection.
- [ ] Results explicitly make no endpoint promotion, scientific convergence,
      native speed win/loss, or formal comparison claim.

## Open Questions

- None. Any change to phase identities, diagnostic thresholds, schedule,
  numerical policy, or artifact schema requires a new SSOT revision and fresh
  preflight; it may not be chosen after observing the cold result.

## Executed Outcome Ledger — 2026-08-11

The production execution was authorized from the immutable pre-execution SSOT
bytes with SHA-256
`e6871072a7011d64e511aa8e8cf7db17d36acedbb33dbbce22b18cd0ae2c6d59`.
This appended ledger changes the current document hash but does not alter or
reinterpret the executed numerical policy.

Contemporaneous session evidence, which is not part of the failed canonical
artifact, recorded green pre-execution qualification on the exact launched
bytes:

- eight isolated `JAX_PLATFORMS=cpu` pytest processes passed
  `52 + 45 + 112 + 61 + 15 + 31 + 30 + 13 = 359` tests;
- Ruff check/format, compileall, `git diff --check`, source hashes, and fresh
  independent core, receipt, runner, and integrated launch-readiness reviews
  passed;
- launched runner SHA-256 was
  `bde4238914353c5bfdf68be0fac1b93c987099f76060b958dd39720d2b0d876c`;
  receipt SHA-256 was
  `54f0b2134e9cd138b80b47e550822726313ae7421285e38611d2f74cd200c53c`;
  optimizer-core SHA-256 was
  `c28a598a56eae109b3e61f846ae58c34b97a2cdc5fe92fdb15af0a668eb380de`;
  NEQ adapter SHA-256 was
  `abf9726e487eb4bda9f82c6092415e988e5a346383c89cec732fe7185b6e6fac`.

The contemporaneous session command record used the frozen RTX UUID, disabled
the compilation cache, forced CUDA/FP64, and invoked `--diagnostic-only` at the
new absent output root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag1-prospective`. Retained child
process evidence proves exactly one lower/compile-only preflight and, only
after that gate passed, exactly one cold child. The retained tree has no warm
path or campaign receipt; the session record reports no tuning, replacement,
or second cold.

The preflight is valid retained evidence:

- producer SHA-256
  `4d101da32a42fd696db7ef8f360db76cccec02c5083325f110a805c560c45aea`;
- status `SUCCESS`, terminal `COMPLETE`, annotated lower/compile-only mode,
  zero Python callbacks, and no solver/finalizer/endpoint-audit dispatch;
- compile interval `296.282090386 s` and process-before-serialization
  `296.599617760 s`;
- exact-child peak GPU memory `4,882,169,856 B` (`0.142791425154108`),
  with policy SHA-256
  `6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99`.

The cold is a retained crash, not a numerical diagnostic:

- child PID `2938142`, process duration `320.501020706 s`, exit code `1`,
  terminal `CRASH`, no producer payload (`cold/producer.json` is canonical
  `{}`, SHA-256
  `ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356`);
- stderr SHA-256
  `30117e0328e23ba016e82b6d4d9b17627d2564093591a612e9a596bbaaf6735a`
  records repeated `CUDA_ERROR_OUT_OF_MEMORY` allocation failures from
  `23.52 GiB` down through `4.36 GiB`, followed by
  `CUDA_ERROR_LAUNCH_FAILED` while synchronizing `loop_result`;
- exact-child memory monitoring retained `2,658` samples and peak
  `4,984,930,304 B` (`0.14579691477290152`). This child-only fraction is not
  total campaign usage.

Read-only live process evidence and the executed source support the causal
inference of a resource collision. The contemporaneously observed long-lived
supervisor held approximately `24.6 GiB` on the same RTX 5090; that parent
amount is not retained in the artifact. The cold child's independently retained
peak was `4,984,930,304 B` (`4.642578125 GiB`). The executed runner's
`_publish_parent_policy_authority()` called the JAX-based fullspace bootstrap,
scaling construction, and `jax.device_get` in the CUDA-configured parent before
spawning either child. JAX preallocation therefore reserved most device memory
in the supervisor; the cold child could not allocate its production working
set. This is an infrastructure/resource-accounting failure, not evidence about
GNTR convergence, the quality predicate, terminal physics, or the proposed
algorithmic next route.

A second independent failure prevented canonical publication. After the crash,
the supervisor wrote `{}` to `cold/producer.json` but constructed an
`ArtifactRef` asserting the typed producer schema. The incomplete-receipt
builder resolved that non-null reference before deriving `COLD_CRASH` and
raised `ValueError: artifact schema differs: cold/producer.json`. Therefore
`diagnostic.json` and `artifact-manifest.json` are absent. The tree is not
canonically sealed: two profiler files remain mode `0664`, and root plus six
descendant directories remain `0755`/`0775`. It must not be chmod'ed,
backfilled, or promoted post hoc.

Retained raw-evidence hashes include:

- cold terminal `46fcb0f8c2b2afb658415b317f8acc1c65a582f2d5c7b4454ceeef42acee3ff5`;
- cold process `0f1027f75a606524fb1436be0f266dbefc3de3afb8db564e961840d418c39111`;
- cold memory `fc306c9b27dbef4c6b5b8a9d7372567424892fd1fa63fbc2f41f57a4e910f6ea`;
- cold memory samples
  `9ccf924ea8310da7223b87ba4f3bb1679bd33b1fdc924911ae45cb7f5312c5ab`;
- Chrome trace `c50e6b6d0566b0e9ecc6d43616292c528ef034a8059c02b24f00cfad329ea12e`;
- XPlane trace `074f982f0ecb0ed83e8f203f8cf5fc1c3276cffea3cccf10591cd7c0d22b064f`;
- source manifest
  `d33001f37fadd3b06d04a1fa3ac6f51075afe9da9c400efe2c3558c9c2ba6cfd`;
- native-reference manifest
  `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.

The truthful adjudication is `DIAGNOSTIC_INCOMPLETE`, failure class
`COLD_CRASH` with allocator-contention OOM evidence. Engineering campaign
receipt, phase-based next-route selection, endpoint/physics diagnosis,
promotion, and formal comparison are all `NOT_PRODUCED`. The one-cold budget
under this SSOT is consumed, so no rerun is authorized. Any successor must use
a new SSOT/schema, keep the supervisor off the GPU (prefer a NumPy or short
CPU-only policy-authority process), require zero parent GPU allocation before
each child, and treat absent producer output as an absent optional reference so
crash evidence can still seal.
