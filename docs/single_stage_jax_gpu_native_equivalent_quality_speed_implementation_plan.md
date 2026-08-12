# Single-Stage JAX GPU Native-Equivalent Quality Speed Plan

**Status:** In progress, Revision 1  
**Last updated:** 2026-08-10  
**Route:** `NEQ-GNTR1`

Revision 1 freezes the timed-candidate versus untimed-audit boundary, the
deterministic derivative-transpose certificate, and the continuation-connected
branch evidence required before the first GPU run. It changes no physics,
optimizer step, acceptance rule, or numerical tolerance.

## Purpose

Define the fair engineering test for the single-stage JAX GPU work. The native
reference stopped after 1,000 BFGS iterations without converging, so full KKT
convergence must not be a prerequisite for comparing GPU and native speed.

This document supersedes the KKT-as-a-speed-prerequisite language in
`docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md`. It does
not alter or reinterpret any closed route, receipt, or artifact.

## Goal Command

```text
/goal Execute docs/single_stage_jax_gpu_native_equivalent_quality_speed_implementation_plan.md as the SSOT through all gated completion criteria. Implement and validate a GPU-native JAX single-stage route that preserves the frozen physics, objective, DOFs, exact constraints, FP64 policy, and continuation-connected branch while intentionally using device-resident control, batching, fusion, and parallel derivative work. Done means a provenance-valid GPU endpoint reaches native-equivalent objective and feasibility with finite certified gradients, and its synchronized time-to-quality is compared with 287.30421751597896 s. Raw KKT is reported as telemetry and is not an engineering speed prerequisite. If the bounded route cannot produce the endpoint, close it with complete evidence.
```

## Goals

- Preserve the exact native physical problem while allowing a different GPU
  optimizer, trajectory, and iteration count.
- Reach the exact native 1,000-iteration objective or better:
  `4.4822246533126125e-08`.
- Produce an endpoint whose Boozer and volume equalities are no worse than the
  reconstructed native endpoint within the frozen FP64 cross-evaluator
  tolerance.
- Require finite FP64 state, objective, constraints, and valid objective
  gradients without requiring KKT stationarity.
- Measure synchronized warm GPU time to the first accepted on-device quality
  candidate, then require its independent endpoint audit to pass before the
  sample is native-equivalent and promotion-eligible.
- Produce a validated engineering disposition even when the route does not
  reach the target.

## Non-Goals

- Proving that the native endpoint or the GPU endpoint is a converged optimum.
- Requiring raw KKT `<= 1e-7` for the engineering speed comparison.
- Matching the native optimizer trajectory, iteration count, or BFGS update
  sequence.
- Reopening or mutating the sealed `CFS-GNTR1` evidence.
- Calling the historical comparison a formal campaign `WIN`; the native
  receipt is partial and non-authoritative.
- Tuning multiple optimizers or running an open-ended parameter sweep.

## Current Context

- The preserved native receipt has SHA-256
  `8118529751f184f60f0c4d26f338cd1832aae579004d62866fb2a2f6617e9fe4`.
  It records `nit=1000`, `normalized_status=budget_exhausted`, outer success
  false, endpoint certificate false, terminal stationary false, and
  constraints satisfied true.
- Its final accepted trajectory row records objective
  `4.4822246533126125e-08` at `287.30421751597896 s`. This timestamp is the
  engineering comparison boundary, not proof of convergence.
- Its final reduced 461-component gradient is finite with infinity norm
  `2.8661774598730246e-05`, about 286.6 times the historical `1e-7`
  stationarity tolerance.
- The native receipt records Boozer residual RMS
  `6.52722677972708e-15`, but its Boolean `constraints_satisfied` does not
  expose a directly comparable 255-component fullspace feasibility norm.
- The closed `CFS-GNTR1` GPU canary preserved feasibility, reduced the
  objective by 21.53%, executed 11 attempts in `12.552584356 s`, and made no
  hot host/device transfers. Its raw KKT worsened, so it was correctly not
  selected by its convergence gate. That result does not answer the present
  time-to-native-quality question.

## Rationale

Performance comparisons must hold result quality fixed, not optimizer
iteration count or a convergence standard that the reference did not meet.
The fair engineering metric is therefore time to the first endpoint with the
same physical objective quality, equivalent exact-constraint quality, and
finite certified derivatives.

`NEQ-GNTR1` is a new bounded route. It reuses the existing device-resident
Gauss-Newton trust-region algorithm and its validated feasibility restoration,
but changes the campaign stopping and adjudication contract: stop when the
native-equivalent quality predicate is met, not when KKT reaches `1e-7`.
KKT remains valuable scientific telemetry and may still be used internally to
construct a constrained search direction; it does not decide the engineering
speed result.

## Assumptions

- The preserved native endpoint can be reconstructed on its
  continuation-connected Boozer branch well enough to produce an
  apples-to-apples fullspace equality reference.
- The existing exact objective ledger, state mapping, FP64 policy, and
  fullspace derivative certificates remain authoritative.
- The current GN trust-region accept/reject, radius, curvature, correction, and
  feasibility policies are reused unchanged; only the bounded loop size,
  quality-stop predicate, and campaign adjudication are new.
- The RTX 5090 is the primary measurement device. Landau A100 measurement is a
  secondary portability result and does not block primary completion.

## Frozen Engineering Quality Contract

The device-resident loop marks an accepted state as a
`DEVICE_QUALITY_CANDIDATE` only when all of the following on-device conditions
hold:

1. It is produced from the frozen problem inputs and policy: identical
   objective terms, signs, weights, targets, fixed first current, free DOFs,
   Boozer and volume equalities, FP64 precision, and orientation conventions.
2. The authoritative physical objective is finite and
   `<= 4.4822246533126125e-08`.
3. Its 255 physical equality components are finite and, component by component,
   no worse than the reconstructed native reference within the existing vector
   parity rule:

   ```text
   abs(q_gpu[i]) <= abs(q_native[i]) + 1e-12 + 1e-10*abs(q_native[i])
   ```

   The independently reported scaled fullspace feasibility must also be
   `<= 1e-10`.
   The accepted-state hook already owns the scaled vector `c_candidate=D*q`.
   It recovers the raw vector without another physics call as
   `q_candidate=c_candidate/D` elementwise; every component of the frozen
   `constraint_inverse_scale` `D` must be finite and nonzero.
4. The corrected accepted physical state, objective, raw equalities, and scaled
   equalities are finite FP64 values, and the correction certificate already
   required by the unchanged GN trust-region route passes.

The timed predicate deliberately uses only values already computed for the
corrected accepted state. It does not perform a second endpoint gradient,
Jacobian, residual, KKT, or native evaluation inside the timed loop.

The candidate becomes `NATIVE_EQUIVALENT_QUALITY` only after the independent
post-timing endpoint audit also proves:

1. the physical state, objective gradient, constraint JVP, and constraint VJP
   are finite FP64 values;
2. the existing objective-residual value defect is `<= 1e-12` and gradient
   defect is `<= 1e-10`;
3. the deterministic constraint JVP/VJP transpose defect defined below is
   `<= 1e-10`;
4. exact objective-term, fixed/free-DOF, state-ordering, and observable parity;
5. continuation connectivity to the common bootstrap Boozer root under the
   frozen replay below; and
6. native-on-GPU and JAX-on-native same-state cross-evaluator agreement.

If this audit fails, the sample is nonpromoting even though its retained device
time remains reportable.

### Deterministic derivative-transpose certificate

At the latched corrected endpoint, in optimizer coordinates, define

```text
z(u) = z0 + S*u
c(u) = D*q(z(u))
```

where `q` is the exact 255-vector ordered as the 254 masked Boozer components
followed by signed-volume-minus-target, `S` is the frozen variable scale, and
`D` is the frozen constraint inverse scale. With one-based indices, use the
fixed FP64 probes

```text
v = sin(arange(1, 717));       v = v / ||v||2
w = cos(0.5*arange(1, 256));   w = w / ||w||2
```

Compute `Jv` with an independent `jax.jvp(c, ...)` and `J^T w` with an
independent `jax.vjp(c, ...)`; neither action may be derived from retained or
materialized rows. Let

```text
a = vdot(w, Jv)
b = vdot(v, J^T*w)
den = max(float64.tiny, ||w||2*||Jv||2 + ||v||2*||J^T*w||2)
transpose_defect = abs(a-b) / den
```

All probes, actions, dots, denominator, and defect must be finite FP64 values.
Exact zero actions define a zero defect. The audit requires
`transpose_defect <= 1e-10`.

### Frozen continuation-connected branch evidence

The preserved native scalar trajectory does not contain coil or inner-state
vectors and therefore cannot prove the historical BFGS seed sequence. This
route instead certifies continuation connectivity to the same canonical
bootstrap branch; it does not claim to reproduce the historical native branch
trajectory.

The native reference is reconstructed along two independent fixed dyadic paths

```text
coil(k) = coil_initial + (k/256)*(coil_native_final - coil_initial),
k = 0, ..., 256.

coil_refined(j) = coil_initial + (j/512)*(coil_native_final - coil_initial),
j = 0, ..., 512.
```

Start from the canonical bootstrap exact root. At each subsequent point seed
the native exact Boozer solve only with the preceding successful
`(surface[253], iota, G)` state. Every solve must use tolerance `1e-13` and at
most `20` Newton iterations and must return finite state and residual evidence,
the frozen residual ordering/orientation, and scaled Boozer feasibility
`<= 1e-10`. At every common knot require

```text
||y_256(k) - y_512(2*k)||inf <= 1e-10,
```

which is the fail-closed refinement check against a Newton basin jump. Require
the terminal 461-vector to equal the retained native vector exactly and require
the reconstructed terminal objective, iota, volume, non-QS term, Boozer
residual value/RMS, major-radius penalty, and length penalty to agree with the
sealed receipt using `rtol=1e-12` and `atol=1e-15`.
Failure at any point yields `REFERENCE_NOT_PRODUCED`; the schedule is not tuned
after observing the result.

For each GPU sample, the device loop retains a fixed optimizer-coordinate
ledger of shape `(257, 716)`: bootstrap `u=0` in row 0 followed only by corrected
accepted states, plus its valid-row mask and accepted-step count. Rejected and
padded rows are never replayed, and the last valid row is the latched endpoint.
After timing, every valid row is mapped exactly as `z=z0+S*u` on device and the
physical ledger is transferred once. Starting from the same
canonical bootstrap root, first require row 0 to pass all finite/residual/order
checks and `||y_gpu(0)-y_bootstrap_exact||inf<=1e-10`. Then set the authoritative
predecessor `y_0=y_bootstrap_exact`. For accepted interval `k`, starting only
from authoritative `y_(k-1)`, compute the direct endpoint root `y_k^d`, the
half-step root `y_(k-1/2)`, and the refined endpoint root `y_k^r`. Require both
paths to pass the same `1e-13`/`20` solve and residual policy and require
`||y_k^d-y_k^r||inf<=1e-10`. Compare the GPU state to `y_k^r`, then set the
authoritative predecessor for interval `k+1` to `y_k^r`. At every row require

```text
||y_gpu(k) - y_native_exact(k)||inf <= 1e-10,
```

where `y=(surface[253], iota, G)`. Record each row's input/output hashes,
predecessor index, residual evidence, state difference, and first failing
index. Any failure is `ENDPOINT_AUDIT_FAILED_NONPROMOTING`.

There is deliberately no gradient-norm or KKT-norm threshold in this
engineering predicate. Raw physical KKT, scaled stationarity, multiplier
diagnostics, and native reduced-gradient norms are recorded as telemetry when
available. Unavailable or nonfinite KKT telemetry is encoded as an explicit
status rather than JSON `NaN`; it does not reject an otherwise valid
engineering endpoint unless the underlying state, objective, constraints, or
objective gradient is itself nonfinite.

## Timing and Verdict Contract

- Warm timing starts after lowering/compilation and device-state construction
  are complete and synchronized, but before the required initial objective,
  constraint, and gradient evaluation for that solve.
- No objective, derivative, or optimizer-state computation for the timed solve
  may be cached from the untimed warm-up.
- Timing ends after the first `DEVICE_QUALITY_CANDIDATE` is synchronized and
  before endpoint audit, host transfer, or serialization.
- The timed executable returns the bounded optimizer-loop result only. Final
  gradient, Jacobian, residual, multiplier/KKT, branch, and cross-evaluator
  work executes through a separate post-timing finalizer. The finalizer may
  certify only the latched state; it may not replace or project that state.
- The candidate decision and early stop execute on device. The hot loop allows
  zero H2D transfers, zero D2H transfers, and zero Python callbacks.
- The post-timing endpoint audit decides whether that candidate is
  `NATIVE_EQUIVALENT_QUALITY`; it cannot retroactively change the recorded
  device time.
- Cold compile, cold synchronized solve, endpoint audit, and total end-to-end
  times are reported separately.
- Exactly one cold and three warm isolated RTX 5090 runs are retained. No sample
  is deleted, replaced, or called an outlier.
- Each warm solve has a 360 s solve timeout; each process has a 900 s timeout.

Terminal engineering dispositions are:

- `ENGINEERING_SPEED_GOAL_ACHIEVED`: all three warm runs independently reach
  `NATIVE_EQUIVALENT_QUALITY`, pass provenance/resource gates, and each timed
  solve is `< 287.30421751597896 s`.
- `ENGINEERING_SPEED_GOAL_NOT_ACHIEVED`: complete valid warm evidence reaches
  native-equivalent quality, but the timing rule above is not met.
- `QUALITY_NOT_REACHED_BOUNDED_NEGATIVE`: the frozen bounded route completes or
  times out without reaching native-equivalent quality.
- `REFERENCE_NOT_PRODUCED`: an apples-to-apples native fullspace feasibility
  reference cannot be reconstructed; no comparative speed disposition is
  issued.
- `ENDPOINT_AUDIT_FAILED_NONPROMOTING`: an on-device candidate was timed, but
  its branch or cross-evaluator audit failed.
- `NOT_PRODUCED`: required source, runtime, endpoint, or receipt evidence is
  incomplete.

Independently, an endpoint may receive the informational label
`SCIENTIFIC_CONVERGENCE_CERTIFIED` only if the existing full KKT and feasibility
certificate passes. That label does not change the engineering timing verdict.
A formal comparative `WIN` or `LOSS` remains `NOT_PRODUCED` until a
claim-compatible native baseline is imported or remeasured.

## Implementation Plan

1. Freeze the native-equivalent reference.
   - [x] Import the preserved native receipt and trajectory by exact path and
         SHA-256; reject any byte drift.
   - [x] Reconstruct its final 461-DOF endpoint through the authoritative native
         evaluator using the frozen independent 256- and 512-segment dyadic
         continuations from the canonical bootstrap branch and require their
         roots to agree at every common knot.
   - [x] Map the reconstructed state into the 716-variable fullspace ordering
         and freeze the 255 physical equality values, state hash, objective
         terms, branch evidence, and cross-evaluator tolerances in a small JSON
         policy artifact.
   - [x] Fail with `REFERENCE_NOT_PRODUCED` if reconstruction, mapping, or
         same-state parity cannot be established. Do not substitute the native
         `constraints_satisfied` Boolean for numerical evidence.

2. Add the device-side time-to-quality contract.
   - [x] Add a route-local immutable quality policy and result type; do not add
         campaign fields to the shared physics evaluator.
   - [x] Extend the GN trust-region loop with an on-device
         `device_quality_candidate_reached` latch and first-hit state/attempt
         counters.
   - [x] Split the timed bounded loop from the untimed finalizer; retain the
         existing public GNTR entry point as a compatibility wrapper.
   - [x] Retain the fixed `(257, 716)` accepted-coordinate ledger and mask needed
         for the post-timing continuation replay.
   - [x] Stop the device loop on quality, fatal numerical evidence, 256 accepted
         steps, or 300 attempted steps, whichever occurs first.
   - [x] Preserve the existing GN model, trust-radius updates, acceptance,
         correction, curvature, and certificate tolerances unchanged.
   - [x] Keep KKT/stationarity values in telemetry but remove them from route
         selection and engineering success predicates.

3. Bind the fullspace adapter and endpoint audit.
   - [x] Build `NEQ-GNTR1` from the canonical bootstrap, scaling, objective
         residual, joint objective/equality evaluator, and endpoint diagnostic
         owners already used by the closed GN route.
   - [x] Recompute the candidate objective, term ledger, 255 equalities, finite
         gradient, residual reconstruction, and JVP/VJP identity independently
         from the optimizer result.
   - [x] Use the frozen deterministic optimizer-space probes and transpose
         formula; do not substitute GN HVP symmetry telemetry.
   - [x] Re-evaluate the qualifying GPU endpoint with the native reduced
         evaluator and report objective, all raw terms, Boozer residual RMS,
         volume, iota, and 461-component reduced-gradient norms.
   - [x] Upgrade the candidate to `NATIVE_EQUIVALENT_QUALITY` only after proving
         the same continuation-connected branch, fixed/free-DOF mapping, and
         two-way same-state cross-evaluator agreement. Do not require KKT
         convergence.

4. Implement the runner, receipt, and validator.
   - [x] Reuse the existing supervised worker, immutable source manifest,
         process GPU-memory monitor, transfer guard, strict canonical JSON, and
         sealed-artifact patterns without modifying closed artifacts.
   - [x] Record the exact native reference hashes, route policy hash, source and
         runtime identities, physical GPU UUID, compile/solve/audit times,
         first quality-hit attempt, objective/equality evidence, gradients,
         KKT telemetry, memory, and transfers.
   - [x] Recompute every numerical and semantic verdict from raw receipt fields;
         never trust producer summary Booleans.
   - [x] Make timeout, crash, compile failure, source mutation, nonfinite data,
         and incomplete evidence produce explicit fail-closed terminal records.

5. Run the bounded RTX 5090 campaign.
   - [x] Run focused CPU/JIT tests and a production-shape lower-only preflight.
   - [x] Run exactly one supervised cold solve. Stop if it cannot produce a
         valid native-equivalent endpoint within the frozen bounds.
   - [ ] After the cold endpoint audit passes, run exactly three isolated warm
         solves from pristine initial state with no replacement. Not executed:
         the cold solve did not reach a device-quality candidate.
   - [x] Validate and seal all produced evidence, then issue exactly one
         engineering disposition from this document's taxonomy.

6. Report the optional A100 result and close the SSOT.
   - [ ] If authorized after the RTX result, run the identical frozen route and
         quality contract on Landau A100 and report it as portability evidence.
   - [x] Update this document with exact commands, hashes, artifacts, timings,
         endpoint values, KKT telemetry, and terminal disposition.
   - [x] Mark completed tasks and leave unexecuted optional work explicit.

## Validation Plan

- [x] Add focused tests for objective equality at the exact threshold, each
      component of the native-relative feasibility rule, nonfinite gradients,
      derivative-certificate failure, and branch mismatch.
- [x] Prove KKT above `1e-7` or explicitly unavailable KKT telemetry does not
      reject an otherwise valid engineering endpoint, and prove nonfinite
      telemetry is represented by a status rather than serialized as JSON
      `NaN`.
- [x] Prove the device loop stops at the first `DEVICE_QUALITY_CANDIDATE` and
      does not execute expensive inactive iterations afterward.
- [x] Prove the timed loop result synchronizes before all finalizer work and
      that the finalizer audits exactly the latched state without projection or
      replacement.
- [x] Prove the accepted-coordinate ledger contains bootstrap plus corrected
      accepted states only, with rejected and padded rows masked.
- [x] Prove the independent 256/512 native-reference continuations agree at
      every common knot, and prove every GPU accepted interval uses the frozen
      direct-versus-midpoint-refined recursion with the refined endpoint as the
      next authoritative predecessor. Fail closed on the first unsuccessful,
      nonfinite, misordered, or branch-mismatched solve.
- [x] Prove a candidate with a failed branch or cross-evaluator audit retains
      its diagnostic time but receives `ENDPOINT_AUDIT_FAILED_NONPROMOTING`.
- [x] Prove compile/warm-up state cannot leak objective, gradients, factors, or
      optimizer state into the timed solve.
- [x] Prove zero hot H2D/D2H transfers and zero Python callbacks.
- [x] Mutate each receipt gate independently and verify fail-closed rejection.
- [x] Run focused test files in separate processes with
      `.venv-qn-gpu/bin/python -m pytest` because JAX platform policy is
      import-time global state.
- [x] Run `ruff check`, `ruff format --check`, `compileall`, and
      `git diff --check` on the owned paths.
- [x] Have an independent reviewer recompute the quality and timing verdict
      directly from the sealed artifacts.

## Risks and Mitigations

- Risk: the historical native endpoint cannot be reconstructed into a directly
  comparable fullspace equality vector.
  Mitigation: terminate as `REFERENCE_NOT_PRODUCED`; never replace it with a
  weaker Boolean or an invented tolerance.
- Risk: changing the stop rule accidentally changes GN numerical steps.
  Mitigation: regression-test the first 8 accepted states against closed
  `CFS-GNTR1` before the new quality predicate can trigger.
- Risk: the objective reaches the target while the coupled surface leaves the
  native branch.
  Mitigation: treat the on-device state only as a candidate until same-state
  cross-evaluator and continuation-connected branch certificates pass.
- Risk: compilation or endpoint audit is hidden inside the warm solve time.
  Mitigation: retain separate synchronized timestamps and validate the event
  ordering in the receipt.
- Risk: a fast but unstable sample is promoted.
  Mitigation: require all three unreplaced warm samples to independently pass
  quality, provenance, resource, and timing gates.
- Risk: the partial native receipt is overstated as a formal baseline.
  Mitigation: use the explicit engineering labels above and keep the formal
  comparative verdict `NOT_PRODUCED`.

## Completion Criteria

- [x] The native reference policy is reconstructed, hashed, and independently
      validated, or the route closes as `REFERENCE_NOT_PRODUCED`.
- [x] `NEQ-GNTR1` executes with a device-resident first-quality stop, unchanged
      physics and GN numerical policy, and zero hot transfers/callbacks.
- [ ] An on-device candidate passes objective, native-relative equality,
      derivative-validity, and FP64 gates, then becomes
      `NATIVE_EQUIVALENT_QUALITY` only after branch and cross-evaluator gates;
      KKT is reported but is not an engineering prerequisite.
- [x] One cold and three warm RTX receipts are complete and independently
      validated, or a truthful bounded-negative/`NOT_PRODUCED` artifact closes
      the route.
- [x] The results document states one terminal engineering disposition and does
      not claim native convergence or a formal comparative `WIN`.
- [x] This SSOT records exact artifact paths, hashes, commands, source/runtime
      identity, timings, endpoint quality, and KKT telemetry.

## Open Questions

- None. Device, route, bounds, quality predicate, timing boundary, and verdict
  taxonomy are frozen by this document. Any change requires a dated SSOT
  revision before another GPU run.

## Executed Outcome — 2026-08-10/11

### Terminal disposition

`QUALITY_NOT_REACHED_BOUNDED_NEGATIVE` for the frozen bounded `NEQ-GNTR1`
route. The route did not trigger `DEVICE_QUALITY_CANDIDATE` within 300 attempted
steps, so no endpoint audit or warm sample was authorized. This does not prove
that GPU JAX cannot beat native under another mathematically valid route or a
larger bound.

The immutable producer artifact itself retains `NOT_PRODUCED`: its cold-receipt
builder used `sha256` instead of the native-reference schema's `file_sha256`
field. That post-run parser defect was fixed and regression-tested without
modifying or rerunning the sealed evidence. Current code independently derives
`QUALITY_NOT_REACHED`, passing provenance/resources, and the bounded-negative
campaign disposition from the retained raw bytes. The distinction between the
immutable producer label and the corrected independent adjudication is
intentional.

No formal comparative `WIN` or `LOSS` was produced.

### Authoritative evidence

- Native reference:
  `artifacts/neq-native-reference-20260811T012049Z`; whole-artifact manifest
  SHA-256
  `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.
- Successful lower/compile-only preflight:
  `/home/jungdaesuh/simsopt-campaigns/neq-gpu-preflight-20260811T0246Z`;
  outer manifest SHA-256
  `78b3a873cf0b3c6cdb8012bcab00a1334bdcde76708940cb847892f7a7ea2386`.
  It recorded one FP64 RTX 5090, zero Python callbacks, no solver/finalizer/audit
  dispatch, 270.696067656 s before serialization, and peak GPU memory
  25,918,701,568 bytes (`0.7580580856871224` of physical memory).
- Cold campaign artifact:
  `/home/jungdaesuh/simsopt-campaigns/neq-gpu-campaign-20260811T0252Z`;
  outer manifest SHA-256
  `1f4a4b99bc22d2080619e43366be1f78b5a9b97efd2319d3b90ead1cfbe7ad6e`.
  The cold producer, terminal, and memory SHA-256 values are respectively
  `16e106212655073a84c6d1ca7d660ec684a34eb3610783291f4110ee474f6df6`,
  `bd3028d59750b8c17e0ad218fa8a19cebf601ad42317863d41c203a8ecd9a704`,
  and `4146f09ebdad519f46f861a6b25ab24c521aa50cdbedd0bae4e0b4b068b9c7e6`.
- Executed source identity: HEAD
  `52dea17ddf3012cf923fc92da78c0d73a17f4625`, snapshot manifest
  `953daa7356ee08e2ad844060a54c89968f1cd445c2a3d30938795caa0710051e`,
  tracked-diff identity
  `7b94ed15291d33090950e58ea1b183e88a60b0274629d63a4ffb40d7b26426ba`,
  and untracked-bytes identity
  `12d745355eb683d5b37ab1555a79a990e25709ce699c441e7c40b62145e2469f`.

### Cold numerical and timing result

- Attempted steps: 300; accepted steps: 203; candidate latch: false.
- Synchronized solve: `296.461143728 s`, which is
  `9.156926212021062 s` (`3.1871882324567%`) slower than the native engineering
  threshold before accounting for the fact that native-equivalent quality was
  not reached.
- Compile: `268.049838599 s`; whole child: `568.916343581 s`.
- Hot H2D transfers: 0; hot D2H transfers: 0; Python callbacks: 0; final
  post-timing D2H: 1.
- Peak GPU memory: 26,029,850,624 bytes
  (`0.761308921397246` of physical memory), from 4,810 exact-PID samples.
- Endpoint objective, 255 equalities, branch audit, cross-evaluator audit, and
  KKT telemetry are unavailable because the device-quality latch never fired.
  This limits causal diagnosis but does not change the bounded no-hit result.

### Commands

The parent working directory was
`/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed`. The literal
preflight command was:

```bash
CUDA_VISIBLE_DEVICES=GPU-7951f78e-c05d-e01c-303f-d644f4341fe1 \
JAX_ENABLE_X64=true \
.venv-qn-gpu/bin/python -m benchmarks.run_single_stage_native_equivalent_quality_campaign \
  --preflight-only \
  --output /home/jungdaesuh/simsopt-campaigns/neq-gpu-preflight-20260811T0246Z \
  --reference artifacts/neq-native-reference-20260811T012049Z \
  --input-root /home/jungdaesuh/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/inputs \
  --interpreter /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-gpu/bin/python
```

The literal cold-campaign command was:

```bash
CUDA_VISIBLE_DEVICES=GPU-7951f78e-c05d-e01c-303f-d644f4341fe1 \
JAX_ENABLE_X64=true \
.venv-qn-gpu/bin/python -m benchmarks.run_single_stage_native_equivalent_quality_campaign \
  --output /home/jungdaesuh/simsopt-campaigns/neq-gpu-campaign-20260811T0252Z \
  --reference artifacts/neq-native-reference-20260811T012049Z \
  --input-root /home/jungdaesuh/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/inputs \
  --interpreter /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-gpu/bin/python
```

In both commands, the working directory resolves the reference argument to
`/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/artifacts/neq-native-reference-20260811T012049Z`.
The isolated child runner independently forced `JAX_PLATFORMS=cuda` and removed
the legacy `JAX_PLATFORM_NAME` setting.

The corrected read-only adjudication is retained at
`artifacts/neq-gpu-campaign-20260811T0252Z-readonly-adjudication.json`, SHA-256
`11884160878c12db5a76a382980a6b4ca9ec64bb847493a1c25fe458b20e6726`.
It binds the immutable campaign manifest, current runner and receipt-module
hashes, and the canonical reconstructed campaign-receipt SHA-256
`804a3268c9dc43d119ad9bfb026a0fa534f6dfc5aea1b0351c30c2af20d23586`.
The exact replay command is:

```bash
.venv-qn-gpu/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np

import benchmarks.run_single_stage_native_equivalent_quality_campaign as runner
from benchmarks.single_stage_native_equivalent_quality_receipt import (
    CampaignReceipt,
    campaign_sha256,
)

root = Path("/home/jungdaesuh/simsopt-campaigns/neq-gpu-campaign-20260811T0252Z")
publication = runner.load_snapshot(root / "source-snapshot")
producer = json.loads((root / "samples/cold/producer.json").read_text())
terminal = json.loads((root / "samples/cold/terminal.json").read_text())
memory = json.loads((root / "samples/cold/gpu-memory.json").read_text())
source = runner._capture_source_identity_evidence(publication, root)
outcome = runner.SupervisedSample(
    runner.SampleName.COLD,
    runner.ChildTerminalStatus.COMPLETE,
    terminal["child_pid"],
    terminal["child_start_time_ticks"],
    terminal["process_seconds"],
    producer,
    memory,
    (),
    source,
    source,
)
inputs = producer["reference_inputs"]
reference = runner.reference_receipt_from_artifact(
    artifact_root=root / "native-reference",
    reference_evidence=runner._artifact_ref(
        root / "native-reference" / runner.REFERENCE_FILENAME,
        root,
        runner.NATIVE_REFERENCE_SCHEMA_VERSION,
    ),
    bootstrap_state=np.asarray(inputs["bootstrap_state"], dtype=np.float64),
    constraint_inverse_scale=np.asarray(
        inputs["constraint_inverse_scale"], dtype=np.float64
    ),
)
sample = runner.build_sample_receipt(
    outcome,
    producer_reference=runner._artifact_ref(
        root / "samples/cold/producer.json", root, runner.WORKER_SCHEMA_VERSION
    ),
    publication=publication,
    campaign_root=root,
    reference=reference,
)
campaign = CampaignReceipt(reference, (sample,))
campaign.validate()
print("sample_quality=" + sample.quality(reference).value)
print(
    "provenance_resources="
    + str(sample.provenance_and_resources_pass(reference)).lower()
)
print("campaign_disposition=" + campaign.disposition().value)
print("campaign_receipt_sha256=" + campaign_sha256(campaign))
PY
```

The exact output is:

```text
sample_quality=QUALITY_NOT_REACHED
provenance_resources=true
campaign_disposition=QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
campaign_receipt_sha256=804a3268c9dc43d119ad9bfb026a0fa534f6dfc5aea1b0351c30c2af20d23586
```

### Validation

- Independent preflight audit: exact 628/628 manifested files, no symlinks or
  mode/hash/size mismatches, exact source/reference manifests, and explicit
  non-dispatch/non-authorization evidence.
- Post-fix focused aggregate: 198 passed; Ruff, compileall, and
  `git diff --check` passed on the NEQ route and evidence owners.
- Read-only replay of the immutable cold bytes: accepted-step count 203,
  `SampleQuality.QUALITY_NOT_REACHED`, provenance/resources true, and
  `EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE`; the retained
  adjudication file and exact replay command above make this post-fix result
  independently reproducible without altering the sealed campaign.
- Optional Landau A100 execution was not run because the RTX route failed the
  cold quality gate.
