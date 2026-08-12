# Single-Stage Changed-State GPU Timeline Implementation Plan

**Status:** Draft
**Last updated:** 2026-08-05

## Purpose

Implement one bounded, provenance-bound measurement that determines where warm,
changed-state time is spent in the native-default custom-JAX single-stage path.
The measurement must distinguish host transfers and line-search control from the
Boozer Newton and exact-adjoint device graph without changing the numerical
problem or reinterpreting the closed r5 speed campaign.

The result is an attribution verdict, not a speed or promotion verdict. It is
intended to choose the next engineering plan:

- `HOST_BOUNDARY_DOMINANT` permits planning a faithful device-resident BFGS
  state machine.
- `NEWTON_ADJOINT_DOMINANT` requires optimizing the measured Newton, dense
  adjoint, or coil-VJP leaf before revisiting the outer optimizer.
- `MIXED` or `UNATTRIBUTABLE` permits neither architectural conclusion.

## Goals

- Produce synchronized GPU traces for the production `jax_gpu_custom` route at
  `native_default` scale, after compilation, on actual changed states.
- Separate and count these phase families: H2D, D2H, host line-search control,
  Boozer Newton residual/JVP work, dense adjoint-matrix construction, LU/solve/
  refinement, adjoint RHS/VJP work, and implicit coil/Biot-Savart VJP work.
- Preserve the current FP64 objective, exact Boozer solve, direct exact-adjoint
  policy, accepted-incumbent semantics, and host L-BFGS-B algorithm.
- Publish raw traces, interval summaries, numerical observations, provenance,
  hashes, and a recomputed fail-closed decision in a new immutable artifact.
- Establish machine-checkable completion and stopping criteria before any GPU
  run is interpreted.

## Non-Goals

- Optimizing or replacing BFGS/L-BFGS-B, Newton, the adjoint, Biot-Savart, or
  any numerical kernel in this plan.
- Introducing FP32, mixed precision, relaxed tolerances, approximate inner
  solves, operator-GMRES substitution, batching line-search candidates, or a
  DESC-style coupled/proximal formulation.
- Editing the frozen r5 protocol, validator, tag, campaign files, or bounded
  closeout.
- Producing `WIN`, `TIE`, or `LOSS`, or claiming causal eliminable wall time
  from overlapping profiler events.
- Comparing the RTX 5090 and A100 numerically as if they were the same runtime.

## Current Context

- Repository snapshot at plan creation:
  `320e5cba814414a43e48cb5b6e53f4ad356a9925`. Existing unrelated untracked
  `.Codex/`, `.claude/`, and documentation files are user-owned and out of
  scope.
- The authoritative closeout is `CLOSED_BOUNDED_NEGATIVE`, `NON_PROMOTING`,
  and `NOT_PRODUCED`; no complete r5 `campaign.json` exists. The observed RTX
  custom/native ratio is diagnostic rather than a frozen-validator verdict.
- `AcceptedIncumbentHostValueAndGrad.value_and_grad` explicitly stages each
  host candidate to the device, calls the compiled evaluator, synchronizes, and
  materializes a scalar and 461-component gradient back to the host.
- The production fast route is
  `SIMSOPT_LBFGSB → minimize_lbfgs_host_core → line_search_value_and_grad_host`.
  It performs sequential direction, default L-BFGS-B line-search, acceptance,
  and history updates between candidate evaluations. More-Thuente belongs only
  to the separate dense-BFGS branch and is outside this measurement.
- The all-device Optax route still measured roughly 9.876 seconds per warm
  value/gradient call in the bounded A100 profile. Host boundaries therefore
  remain a hypothesis, not a proven dominant cause.
- `simsopt_jax.runtime.host_boundary` is the existing SSOT for explicit D2H
  materialization and already provides context-local transfer call/leaf/byte
  accounting. It does not measure DMA duration and must not be duplicated.
- The active exact-Newton route is matrix-free/JVP-based during its hot loop.
  It must be labeled `newton_residual_jvp`, not `dense_newton_jacobian`.
- The direct exact-adjoint route conditionally materializes a dense transpose
  operator by chunked basis-vector sweeps, performs one LU factorization, uses
  `lu_solve`, applies refinement, and runs safety checks. Factor and batched-RHS
  reuse already exist in source; this measurement must verify runtime counts
  before proposing additional reuse.
- The public Biot-Savart VJP has its own cached JIT, but traceable objective AD
  can inline equivalent VJP work. Attribution must follow trace scopes rather
  than assume every coil derivative calls the public wrapper.

## Rationale

JAX dispatch is asynchronous, and internal device phases are fused or nested.
Adding `perf_counter` calls around Python functions or synchronizing after each
device subphase would serialize the workload and measure the instrumentation.
The selected design therefore combines:

1. existing host-boundary counters and monotonic host intervals;
2. `jax.profiler.TraceAnnotation`/`StepTraceAnnotation` for host evaluation and
   accepted-iteration envelopes;
3. `jax.named_scope` metadata at the existing numerical owners so XLA/CUDA
   work remains inside the original compiled graph; and
4. an exported profiler trace whose GPU kernel/memcpy intervals are assigned to
   the deepest recognized scope and unioned before shares are computed.

Raw nested event durations may exceed wall time. The validator therefore works
from interval unions and preserves an explicit unattributed share. It never
sums nested durations and calls the result eliminable wall time.

### Design classification and alternatives

The anticipated implementation is **Tier 2**: it adds an internal annotation
owner plus benchmark/receipt/validator modules and annotates several existing
numerical owners.

- **Chosen:** one internal trace-annotation SSOT, source-local named scopes, and
  a benchmark-owned runner/receipt/validator. Numerical signatures and public
  APIs remain unchanged.
- **Rejected:** synchronize after every Newton/adjoint subphase. This changes
  scheduling and cannot measure the production graph faithfully.
- **Rejected:** pass a profiler object or boolean through every numerical
  function. That leaks benchmark policy across the solver API and creates
  recompilation/static-argument variants.
- **Rejected:** infer internal phase time from Python call duration alone. JAX
  launches asynchronously, so those durations do not establish device work.
- **Rejected:** append fields to the closed r5 receipt. This study has a new
  identity, schema, root, and verdict vocabulary.

## Assumptions

- The first authoritative run uses one pinned RTX 5090 host, one GPU, FP64, the
  current direct exact-adjoint route, and the native-default input bundle. An
  A100 repetition is optional and must be a separate artifact and verdict.
- The installed JAX/jaxlib combination emits an exported profiler trace that
  contains host annotations, XLA scope metadata, CUDA kernels, and memcpy
  events. The implementation preflight must prove this on a small non-claim
  probe before the production-shaped run.
- The custom route can complete at least seven accepted changed-state outer
  iterations inside a bounded diagnostic child. Failure to do so is a runtime
  result, not permission to reduce physics resolution or change the solver.
- Instrumented and uninstrumented controls use fresh processes, identical
  input/configuration identities, separate cache directories, and the same
  preallocation/transfer-guard policy.
- No new Python dependency is required. Trace parsing is version-bound to the
  captured JAX/jaxlib trace schema and fails closed on an unsupported schema.

## Phase Taxonomy and Ownership

The phase IDs are immutable schema values owned once by the new internal trace
module. Source sites import them; they do not repeat free-form strings.

| Phase ID | Meaning | Primary source seam |
|---|---|---|
| `host.h2d_submit` | Explicit candidate staging; host submission span, not assumed DMA time | `surface_objectives_traceable.py::AcceptedIncumbentHostValueAndGrad.value_and_grad` |
| `host.line_search_control` | Host work after one synchronized evaluation and before the next H2D submission | `optimizer_host_lbfgs.py` |
| `optimizer.lifecycle` | Correlate initial/trial/final evaluations with post-evaluation acceptance | `examples/jax/parity/cases/native_boozerqa.py` and the timeline runner |
| `newton.warm_start` | Predictor/accepted-anchor warm-start work | `surface_objectives_traceable.py` |
| `newton.residual_jvp` | Exact Boozer residual and matrix-free Newton JVP work | `boozer_surface.py`, `optimizer.py` |
| `newton.linear_solve` | Newton linear solution/refinement work | `optimizer.py`, `linear_solve.py` |
| `adjoint.outer_vjp_rhs` | Direct outer VJP and adjoint RHS construction | `surface_objectives_traceable.py` |
| `adjoint.dense_matrix` | Chunked basis sweep that materializes the transpose operator | `linear_solve.py::_dense_square_operator_matrix` |
| `adjoint.lu_factor` | Direct FP64 LU factorization | `linear_solve.py::_solve_dense_square_operator_lu_system_with_status` |
| `adjoint.lu_solve` | Initial and batched RHS solves | same direct-LU owner |
| `adjoint.refinement` | Correction solve, residual, and safety checks | same direct-LU owner |
| `adjoint.implicit_coil_vjp` | Implicit stationarity derivative through coils | `surface_objectives_traceable.py` |
| `biotsavart.forward` | Field evaluation kernels reached by the traceable objective | `biotsavart.py` |
| `biotsavart.vjp` | Field/coil reverse kernels, including inlined equivalents | `biotsavart.py` and enclosing traceable VJP scope |
| `host.d2h_materialize` | Synchronized scalar/gradient materialization | `surface_objectives_traceable.py::AcceptedIncumbentHostValueAndGrad.value_and_grad` |

`biotsavart.*` may be nested under Newton or adjoint phases. It is reported as
a leaf drill-down and is not added again to its parent’s wall share.

## Implementation Plan

1. Freeze the diagnostic contract and protect the closed campaign.
   - [ ] Record HEAD, `git status --short`, branch, Python/JAX/jaxlib, CUDA
     runtime/driver, device name/UUID, CPU identity, affinity, relevant XLA/JAX
     environment, and exact executed-source hashes before implementation and
     again before every authoritative child.
   - [ ] Define a new artifact ID and schema:
     `single-stage-changed-state-gpu-timeline-v1`.
   - [ ] Add a static test proving the new runner cannot write into the r5
     campaign root and imports neither the frozen r5 validator nor receipt
     writer as an extension surface.
   - [ ] Keep the six frozen r5 files byte-identical and verify them using the
     existing frozen-file mechanism before and after this work.

2. Add one internal annotation owner without changing numerical interfaces.
   - [ ] Add `src/simsopt_jax/runtime/trace_annotations.py` containing the typed
     phase IDs and the minimal device-scope and host-annotation helpers.
   - [ ] Implement device scopes with `jax.named_scope`; do not add callbacks,
     transfers, arrays, solver branches, dynamic imports, or profiler flags to
     numerical function signatures.
   - [ ] Implement host spans with the current JAX profiler annotation API and
     `time.perf_counter_ns`; timestamps are diagnostic metadata and remain
     outside jitted computation.
   - [ ] Extend the existing `HostTransferAudit` only to correlate transfer
     counts/bytes with evaluation IDs. Keep D2H semantics owned by
     `runtime/host_boundary.py`; do not create a second materialization ledger.
   - [ ] Unit-test nesting, exception-safe context cleanup, context-local
     isolation, immutable emitted records, and zero numerical outputs from the
     annotation layer.

3. Annotate the real host boundary and line-search control path.
   - [ ] In `AcceptedIncumbentHostValueAndGrad.value_and_grad`, emit ordered
     `host.h2d_submit`, device-candidate envelope, completion barrier, and
     `host.d2h_materialize` events keyed by evaluation index and canonical
     parameter SHA-256.
   - [ ] Record submission/materialization wall spans honestly. Call them CUDA
     memcpy time only when a matching profiler memcpy event corroborates them.
   - [ ] In `optimizer_host_lbfgs.py`, emit the exclusive
     `host.line_search_control` interval between the previous evaluator return
     and the next evaluator call. Exclude evaluator time from this interval.
   - [ ] Receipt-bind the exact production route identity:
     `SIMSOPT_LBFGSB`, `minimize_lbfgs_host_core`, and its default
     `line_search_value_and_grad_host`. Reject More-Thuente or dense BFGS in an
     artifact declaring the production custom route.
   - [ ] Let the canonical case/runner integration own lifecycle correlation.
     Join evaluator IDs and parameter SHA-256 values to the post-evaluation
     `incumbent_controller.accept` callback to assign accepted/rejected status
     without evaluating the objective again.
   - [ ] Classify the required initial evaluation as `initial`, optimizer
     candidates as `trial`, and the post-window
     `session.evaluate_candidate_from_anchor` call as `final_reporting`.
     Exclude `final_reporting` from the seven-iteration attribution denominator
     while retaining its numerical evidence and hash.
   - [ ] Extend the existing fake-evaluator and transfer-audit tests to prove
     H2D → device envelope → readiness → D2H order and exactly one correlated
     host event set per objective/gradient evaluation.
   - [ ] Add a production-route regression proving every evaluation maps exactly
     once to `initial`, `trial`, or `final_reporting`, every trial receives one
     accepted/rejected disposition, and lifecycle correlation performs no extra
     objective evaluation.

4. Annotate the compiled Boozer/adjoint graph at existing owners.
   - [ ] Scope exact residual and JVP work separately from the Newton linear
     solve. Do not label matrix-free Newton as dense-Jacobian construction.
   - [ ] Scope outer VJP/RHS creation, dense transpose materialization, LU
     factorization, each LU solve, refinement/safety work, and implicit coil VJP
     at their current function owners.
   - [ ] Scope Biot-Savart forward and reverse kernels at the shared kernel
     owner while retaining the enclosing objective/adjoint scope for inlined
     reverse work.
   - [ ] Emit counts for Newton iterations, dense materializations, LU
     factorizations, LU solves, refinement corrections, and Biot-Savart forward/
     reverse kernel groups. Counts describe execution; no expected count is
     invented before observing the route.
   - [ ] Add unit tests that compare annotated and unannotated values, gradients,
     statuses, residual traces, and cache behavior on deterministic CPU
     fixtures.

5. Implement the bounded changed-state trace runner.
   - [ ] Add `benchmarks/run_single_stage_changed_state_gpu_timeline.py`; reuse
     canonical native-default construction, the accepted-incumbent controller,
     existing provenance extraction, and existing process/GPU preflight owners.
   - [ ] Warm compilation before starting the profiler and verify that no
     compilation event occurs inside a warm measurement envelope.
   - [ ] Run the receipt-bound production custom host L-BFGS-B/default-line-search
     route for exactly seven accepted outer iterations, retaining every
     intervening line-search trial.
     Every non-initial candidate must have a distinct parameter SHA-256.
   - [ ] Use `StepTraceAnnotation` for accepted iterations and
     `TraceAnnotation` for trial/evaluation envelopes. Capture the JAX profiler
     trace in a fresh, non-`/tmp`, non-overwriting artifact root.
   - [ ] Run three independent profiled children and three independent
     uninstrumented control children in alternating order. Each child receives
     its own compilation-cache directory; environment and cache policy are
     otherwise identical and receipt-bound.
   - [ ] Bound every child by wall time and process-tree memory. A timeout,
     nonfinite gradient, failed inner solve, missing accepted iteration, or
     incomplete trace is serialized as diagnostic failure and is never filled
     with synthetic timing.

6. Add a separate receipt, trace summarizer, and validator.
   - [ ] Add `benchmarks/single_stage_changed_state_gpu_timeline_receipt.py` to
     own typed records, canonical JSON/JSONL, exclusive file creation, hashes,
     manifest roles, staging, fsync, validation-before-publication, and atomic
     directory rename.
   - [ ] Add `benchmarks/summarize_single_stage_changed_state_gpu_timeline.py`
     to parse the exact captured trace schema. Reject unknown trace schemas;
     do not heuristically guess event names.
   - [ ] Assign a CUDA kernel/memcpy interval only when its trace metadata has one
     unique deepest recognized phase scope. Reject ambiguous equally specific
     or multi-owner scopes into `unattributed_device`; never choose one by source
     order. Union intervals before computing active time. Preserve raw nested
     sums, overlap, and unattributed device time as separate diagnostic fields.
   - [ ] Derive host control gaps only from correlated evaluator-return and
     next-evaluator-entry events. Reject missing, reversed, duplicated, or
     cross-evaluation correlations.
   - [ ] Add `benchmarks/validate_single_stage_changed_state_gpu_timeline.py`.
     It independently recomputes hashes, interval unions, per-iteration shares,
     per-process medians, numerical equivalence, coverage, and the final verdict
     from raw evidence; `decision.json` is output, never trusted input.
   - [ ] Publish the validator's terminal result atomically in a fresh sibling
     `<artifact-root>.validation/validation_result.json`. For an invalid root,
     this external validation result is authoritative and no `decision.json`
     inside the untrusted root is used or created.
   - [ ] Require the manifest to bind every claim-bearing file by role, relative
     path, size, SHA-256, source state, process, and sample/evaluation ID.

7. Prove instrumentation does not change the numerical run.
   - [ ] For every profiled/control pair, require identical input,
     configuration, construction, source, runtime policy, initial parameter,
     and driver identities.
   - [ ] Compare accepted objective sequence, accepted parameter hashes,
     `nit/nfev/njev`, line-search decisions, status, final parameters, iota,
     volume, non-QS ratio, Boozer residual, inner-solve evidence, and endpoint
     certificate using the existing frozen parity tolerances.
   - [ ] Require finite FP64 values and gradients and candidate-sourced gradients
     for every attributed changed-state evaluation.
   - [ ] Report profiler overhead as the paired profiled/control wall ratio.
     Profiler-event durations are structural-only if the median overhead exceeds
     10%; such a run cannot receive a dominance verdict.

8. Execute the decision rule and stop at the selected branch.
   - [ ] Compute, for every accepted iteration, mutually exclusive critical-path
     shares for `host_boundary`, `newton_adjoint`, `other_attributed`, and
     `unattributed`. Nested Biot-Savart leaves are drill-down only.
   - [ ] Require all three profiled children to be valid, each with seven
     accepted changed-state iterations, and median unattributed share no greater
     than 20%.
   - [ ] Emit `HOST_BOUNDARY_DOMINANT` only when the pooled median host share is
     at least 60%, every process median is at least 50%, and the pooled host
     share exceeds Newton/adjoint by at least 10 percentage points.
   - [ ] Emit `NEWTON_ADJOINT_DOMINANT` only under the symmetric 60%/50%/10-point
     rule.
   - [ ] Emit `MIXED` when evidence is valid and attributable but neither
     mechanism passes its dominance rule.
   - [ ] Emit `UNATTRIBUTABLE` when trace coverage, clock correlation, profiler
     overhead, or the unattributed-share gate fails. Emit `SCIENTIFIC_INVALID`
     or `INTEGRITY_ERROR` before attribution when those gates fail.
   - [ ] For `HOST_BOUNDARY_DOMINANT`, write a separate device-resident L-BFGS-B
     implementation plan preserving the production default line-search,
     accepted-incumbent, and status semantics. Do not implement it inside this
     work.
   - [ ] For `NEWTON_ADJOINT_DOMINANT`, select the measured leaf: verify active
     scalar-adjoint and factor-reuse counts before proposing restructuring;
     choose dense assembly/LU, Newton JVP, or coil-VJP batching/fusion according
     to the trace. Do not assume reuse is absent.
   - [ ] For every other verdict, stop without an optimization plan until the
     attribution defect or mixed cost structure is resolved.

## Validation Plan

Run JAX-focused pytest files in separate processes because import-time platform
configuration can leak between modules.

```bash
env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/test_runtime_host_boundary.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/test_host_boundary_ssot_ratchet.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/geo/test_traceable_trial_evaluator.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/geo/test_adjoint_cg_solver.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/field/test_biotsavart_jax.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/benchmarks/test_boozer_trial_diagnostic.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/benchmarks/test_custom_quasi_newton_runtime.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/benchmarks/test_single_stage_changed_state_gpu_timeline.py -q

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest tests/benchmarks/test_validate_single_stage_changed_state_gpu_timeline.py -q
```

The new synthetic receipt/validator tests must cover:

- [ ] Every valid terminal verdict.
- [ ] Unknown schema and phase IDs.
- [ ] Missing, duplicated, reversed, or overlapping intervals in host spans
  declared exclusive, and duplicate assignment of one CUDA interval to multiple
  leaf owners.
- [ ] Correct interval-union behavior for legitimate nested/overlapping GPU
  events, including ambiguous multi-owner kernels becoming unattributed.
- [ ] Missing synchronization or completion events.
- [ ] Trace schema drift and absent CUDA memcpy/kernel records.
- [ ] Parameter/evaluation/iteration correlation mismatch.
- [ ] Source, environment, device, cache, manifest, trace, trajectory, and
  numerical-observation tampering.
- [ ] More than 10% profiler overhead and more than 20% unattributed share.
- [ ] Runtime route selecting a non-direct adjoint despite the declared policy.
- [ ] A failed or incomplete child remaining diagnostic instead of receiving an
  attribution verdict.

Static and formatting checks:

```bash
.venv-qn-gpu/bin/python -m ruff check \
  benchmarks src/simsopt_jax tests/benchmarks tests/geo tests/field
.venv-qn-gpu/bin/python -m ruff format --check \
  benchmarks src/simsopt_jax tests/benchmarks tests/geo tests/field
.venv-qn-gpu/bin/python -m compileall -q benchmarks src/simsopt_jax tests
git diff --check
```

Authoritative GPU run, after CPU/synthetic validation passes:

```bash
TIMELINE_ROOT=/home/jungdaesuh/simsopt-campaigns/single-stage-changed-state-gpu-timeline-v1
env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python \
  benchmarks/run_single_stage_changed_state_gpu_timeline.py \
  --case native-single-stage-boozer-vacuum-optimization \
  --scale native_default \
  --accepted-iterations 7 \
  --profile-children 3 \
  --control-children 3 \
  --gpu-index 0 \
  --artifact-root "$TIMELINE_ROOT"

.venv-qn-gpu/bin/python \
  benchmarks/validate_single_stage_changed_state_gpu_timeline.py \
  "$TIMELINE_ROOT"
```

## Risks and Mitigations

- Risk: named scopes do not survive lowering/fusion in a form that uniquely
  identifies CUDA intervals.
  Mitigation: require a minimal trace-schema preflight and emit
  `UNATTRIBUTABLE` rather than inferring ownership from source order; a fused
  kernel with multiple equally specific owners is unattributed.
- Risk: annotations or profiler collection perturb scheduling.
  Mitigation: rotate three profiled/control pairs, compare numerical paths, and
  reject quantitative attribution above the 10% overhead gate.
- Risk: CUDA memcpy events and host submission spans use different clock
  domains or cannot be correlated.
  Mitigation: preserve both, validate clock metadata, and label unmatched time
  unattributed; never rename a host submission span as DMA time.
- Risk: nested Biot-Savart work is counted both under Newton/adjoint and as a
  separate phase.
  Mitigation: assign kernels to the deepest recognized leaf, union intervals,
  and treat Biot-Savart totals as drill-down rather than additional wall share.
- Risk: a seven-iteration diagnostic is dominated by initial compilation or a
  same-point cache.
  Mitigation: warm before tracing, reject compile events in warm envelopes, and
  require distinct non-initial parameter hashes.
- Risk: instrumentation work creates a parallel profiling or transfer SSOT.
  Mitigation: keep D2H accounting in `runtime/host_boundary.py`, phase IDs in
  one trace module, and artifact policy in one receipt/validator family.
- Risk: a trace labels source structure but not causal eliminable time.
  Mitigation: word the result as attribution only; require a separate faithful
  implementation/canary plan before claiming a speed improvement.

## Completion Criteria

Implementation is complete only when the instrumentation, artifact, and
validator tests below pass. The measurement attempt is terminal when exactly
one row of the following table applies. Failure rows are deliberately allowed
to close without fabricated successful children.

| Terminal verdict | Machine-checkable minimum evidence |
|---|---|
| `HOST_BOUNDARY_DOMINANT` | Three valid profiled and three valid control children; seven accepted changed-state iterations per child; scientific equivalence; profiler overhead ≤10%; unattributed share ≤20%; all required host/device phase families present; 60%/50%/10-point host rule passes |
| `NEWTON_ADJOINT_DOMINANT` | Same valid-evidence gates as above; symmetric Newton/adjoint dominance rule passes |
| `MIXED` | Same valid-evidence gates as above; neither dominance rule passes |
| `UNATTRIBUTABLE` | Source/runtime identity and raw trace bytes are valid, numerical control is valid when execution reached it, but trace-schema preflight, unique scope assignment, clock correlation, profiler-overhead, phase-presence, or unattributed-share gate fails; exact failing gate is serialized and no dominance shares are published |
| `SCIENTIFIC_INVALID` | Source/runtime identity is valid and the first numerical/inner/adjoint/trajectory failure is serialized with its evaluation ID and available raw trace; complete children and attribution coverage are not required |
| `INTEGRITY_ERROR` | The independent validator records the exact missing, malformed, drifted, or hash-invalid evidence and exits nonzero; no attribution or scientific conclusion is published from the invalid root |

This plan is **Done** only when all common implementation/publication criteria
and the selected row-specific criteria are true:

- [ ] The annotation owner, host-boundary/line-search instrumentation, device
  scopes, runner, receipt, summarizer, validator, and focused tests are present
  and pass their validation commands.
- [ ] Default, non-profiled numerical APIs and optimizer behavior are unchanged;
  no new public API, solver option, precision mode, or numerical branch exists.
- [ ] The six frozen r5 files remain byte-identical, and the new artifact uses a
  distinct non-overwriting root and schema.
- [ ] For `HOST_BOUNDARY_DOMINANT`, `NEWTON_ADJOINT_DOMINANT`, or `MIXED`, three
  complete profiled children and three complete controls exist from one pinned
  source/runtime/device state, with seven accepted changed-state iterations per
  child. Failure verdicts satisfy their table row instead.
- [ ] Every evaluation used in a dominance or mixed verdict has finite FP64
  objective/gradient, successful inner/adjoint evidence, canonical parameter
  identity, and complete trace/numerical/provenance/hash binding. A scientific
  failure verdict instead binds the first failing evaluation and available
  evidence.
- [ ] For a dominance or mixed verdict, profiled and control trajectories,
  line-search decisions, endpoints, and observables pass unchanged existing
  tolerances. `UNATTRIBUTABLE` requires the same numerical control when the
  trace reached executable measurement; `SCIENTIFIC_INVALID` records why it
  could not pass.
- [ ] The validator recomputes interval unions, counts, shares, profiler
  overhead, unattributed time, process medians, and the verdict from raw bytes.
- [ ] The 10% profiler-overhead, 20% unattributed-share, and required-phase gates
  apply to dominance and mixed verdicts. `UNATTRIBUTABLE` records the exact gate
  that failed rather than pretending it passed.
- [ ] Exactly one fail-closed terminal validation result is published:
  `HOST_BOUNDARY_DOMINANT`, `NEWTON_ADJOINT_DOMINANT`, `MIXED`,
  `UNATTRIBUTABLE`, `SCIENTIFIC_INVALID`, or `INTEGRITY_ERROR`.
- [ ] The result document states confirmed timings, descriptive attribution,
  inference, and unmeasured/causal claims separately.
- [ ] The next action is limited to the branch authorized by that verdict; no
  optimizer or numerical optimization is implemented as part of this plan.
- [ ] A fresh independent review verifies the final diff, raw artifact/manifest
  hashes, validator recomputation, claim ceiling, and workspace status.

## Open Questions

- Does the selected JAX/jaxlib trace export preserve enough `jax.named_scope`
  metadata after fusion to assign CUDA kernels uniquely? Phase 1 preflight owns
  this answer; failure yields `UNATTRIBUTABLE`, not a substitute timer design.
- Which exact device host will own the first authoritative run? Default:
  the local RTX 5090 used for the preserved diagnostic. Any A100 run is a
  separately identified portability artifact.
- Does the production route execute one dense materialization/factorization per
  changed-state gradient, or more through nested objective terms and safety
  rebuilds? The new execution counts must answer this before factor-reuse work
  is proposed.
