# Custom JAX BFGS/L-BFGS Step Runtime Implementation Plan

**Status:** In progress — core runtime green; physics and promotion gates open

**Last updated:** 2026-08-01

**Change tier:** Tier 3 — public solver behavior and traced call paths

## Execution status (2026-08-01)

- Green: typed eager runtime, BFGS eager steps, L-BFGS dynamic-budget facade,
  and SciPy Rosenbrock accepted-step parity. The private wrapper cache now has
  capacity-8 LRU/single-flight admission with tests for owner lifecycle,
  eviction of an in-use wrapper, key identity, failed compilation, cardinality,
  and concurrent first use. Tree-definition key separation is covered for
  structured adapters; full adapter execution collision coverage and XLA
  executable reclamation are not yet qualified.
- Green: CPU runner records fixture-build elapsed time and the RSS observed at
  the build boundary separately from cold/warm solver time, provider-child
  RSS, status, counters, full initial/final parameter vectors, and explicit
  Optax comparison on deterministic contract fixtures. Each measurement now
  also records generator/source hashes, full fixture-contract metadata,
  expected initial observables, solver options, predeclared tolerances, and
  final certificate fields. The measurement payload is now schema version 5;
  version 3 adds the phase-separated warm transfer audit, version 4 adds
  solver-boundary RSS start, peak, and delta fields, and version 5 records the
  JAX/XLA/SIMSOPT runtime environment.
  Commit/dirty-state and dependency versions remain recorded; the supported
  Rosenbrock command was re-run from the documented environment.
- Green: schema-4 matched CPU `coil47` receipts now expose solver-boundary
  memory. Native/custom/Optax solver RSS deltas were `0 / 355824 / 1699508`
  KiB; final objectives remained `0.13786263284430203 / 0.137862632844302 /
  0.137862632844302`. Receipts are under
  `.artifacts/custom-quasi-newton/20260801T-coil47-{native,custom,optax}-cpu-schema4-fixed/`.
  This is CPU evidence only; strict GPU and A100 lanes remain open.
- Green: provider children now have direct-PID 120-second and 8-GiB RSS
  fail-closed watchdogs with TERM/KILL grace; both limit paths have bounded
  contract tests without allocating an 8-GiB fixture.
- Green: eager BFGS now preserves nonfinite/failed initial-stop semantics and
  maps callback `StopIteration` to unsuccessful status 99 through the typed
  result boundary.
- Green: traced whole-solve BFGS is reachable again; tracer closure constants
  now remain under the enclosing JIT's placement instead of probing tracer
  shardings.
- Green: traceable Newton refinement now normalizes optional linear-solve status
  fields before `lax.cond`; the previously failing six-case traceable private
  compatibility slice passes.
- Green: traceable Boozer penalty objective and residual closures are now
  cached per instance by the existing structural key. Repeated LM/Boozer
  calls reuse the same callables; target-label and option changes still build
  fresh closures. The RED regression was caused by rebuilding the objective
  on every call and is now covered by stable/rebuild tests.
- Green: eager BFGS and stepwise L-BFGS observer paths pack accepted-step
  payloads at their explicit host boundary; monolithic compatibility callbacks
  still use their legacy boundary and remain separately qualified.
- Green: dense-BFGS receipts now include a conservative logical memory contract
  with old/new Hessian bytes, update intermediates, derived peak-live upper
  bound, and explicit no-donation policy. This is accounting, not measured
  device memory.
- Diagnostic: the bounded CPU Boozer BFGS receipt (65 variables, two-step cap)
  measured a custom solver-boundary RSS delta of `1,521,644 KiB` versus native
  `0 KiB`; the custom logical upper bound is only `204,880` bytes. This is
  process-level compile/allocator evidence, not proof of dense-update device
  bytes, and both endpoints are capped (`status=1`, `success=false`).
- Green: L-BFGS eager budget scalars now use the device-placement owner rather
  than raw host `jnp.asarray`; the explicit value/gradient transfer-guard test
  and the fast public BFGS/L-BFGS CPU subset pass after the fix.
- Decision: use physical design B. An exploratory monolithic L-BFGS probe
  exceeded 120 seconds and 5 GiB RSS; an exploratory specialized-kernel facade
  completed the same Rosenbrock probe in about 3.9 seconds cold and 0.6 GiB
  RSS. These figures are not promotion receipts.
- Open diagnostic: an earlier repaired compile-shape attempt produced no local
  receipt. The bounded CPU probe now writes a receipt, but the full root-cause
  matrix is still open. The default physical design-B kernels compile five
  executables with no recompilation across three short-budget calls; the
  short probe records a finite capped result (`status=1`, two iterations)
  instead of incorrectly aborting on `converged=False`. A small Boozer
  compile-log probe likewise records five stepwise executables and no
  monolithic compile. The full legacy-kernel compile reached 9,420,664 KiB RSS
  after about 4 minutes 11 seconds before manual interruption; legacy kernels
  remain explicit opt-in.
- Safety finding: an exploratory unbounded attempt to combine the four root-
  cause cells in one process reached 6.1 GiB RSS while compiling the
  `coil47/maxcor=300` cell and was manually terminated without a receipt. The
  matrix must use the existing direct-PID watchdog before it can produce
  authority evidence; no result from that attempt is promoted.
- Diagnostic receipt: the separately bounded `coil47`, `maxcor=10`, compile-only
  lowering probe reached the 120-second wall limit at `3,325,424 KiB` direct-child
  RSS under a 4-GiB cap and exited by watchdog (`-15`) without a compile payload.
  Receipt: `.artifacts/lbfgs-ondevice/root-cause-coil47-maxcor10-20260801.json`.
  This isolates a compile/objective-graph bottleneck before runtime warm timing;
  it is not evidence of a solver failure and is not promoted.
- Green scaffolding: `benchmarks/lbfgs_compile_root_cause_matrix.py` now runs
  the five declared CPU cells in independent direct-PID children, samples RSS,
  and records completed, timeout, RSS-limit, or failed outcomes. Its contract
  tests cover the cheap/coil, `maxcor=10/300`, short/long, and compile-only/warm
  axes. The exact-watchdog run now has durable receipts, but no matrix result
  is promoted until the required payload fields are produced.
- Diagnostic receipt: the guarded cheap quadratic `maxcor=300`, `maxiter=20`
  warm cell timed out at `120.38 s` with direct-child peak RSS
  `4,199,576 KiB`; receipt:
  `.artifacts/lbfgs-ondevice/root-cause-quadratic-maxcor300-20260801.json`.
  It wrote no compile-shape payload. The result confirms the watchdog is operating and that the
  history-size/compile interaction needs isolation; it is not a solver
  correctness or promotion result.
- Diagnostic matrix result: all five declared cells were rerun under the same
  120-second/8-GiB policy and timed out without payloads. Peak direct-child RSS
  was `4,848,656 / 4,199,576 / 3,322,576 / 3,327,304 / 4,562,308 KiB` for
  quadratic-10, quadratic-300, coil47-10 compile, coil47-10 warm, and
  coil47-300 compile, respectively. The matrix therefore confirms bounded
  failure behavior but does not yet provide compile events, StableHLO sizes,
  executable counts, or iteration timings. Receipts are under
  `.artifacts/lbfgs-ondevice/`.
- Historical probe: the earlier source-owned Boozer custom route used a
  different inner root (`iota=-0.05134074584230428`) and is not parity
  evidence. Its receipt remains archived under
  `.artifacts/custom-quasi-newton/20260801T0720Z-cpu-boozer-custom/`.
- RED -> GREEN: the Boozer fixture had passed
  `constraint_weight=11.1232` to the JAX exact inner solve, unlike the native
  reference. The regression failed on the old fixture, then passed after the
  option was removed and a SIMSOPT-native `BoozerSurface` objective callback
  was added. The corrected initial inner state is
  `iota=-0.1924157185150927`, `G=14.035365807510038`.
- Corrected Boozer CPU callback probe (`maxiter=2`) records native/JAX initial
  objective `3.902843220850033e-4` versus
  `3.902843220850035e-4` (absolute difference
  `2.168404344971009e-19`) and initial gradient-infinity norms
  `3.8700625919864456e-3` versus `3.8700625919863463e-3` (difference
  `9.93129189996722e-17`). Native/custom cold/warm times were
  `2.257643948076293/2.2763357399962842 s` and
  `69.86358919506893/0.038925772067159414 s`; child RSS was
  `970348/3268868 KiB`. The two BFGS implementations took different capped
  endpoints (`2.707807798339409e-4` versus `3.279124574199254e-4`), so this
  receipt certifies initial objective/gradient alignment only, not endpoint or
  convergence parity.
- Probe: matched native SIMSOPT/simsoptpp `BiotSavart` + `SquaredFlux` versus
  custom JAX on source-owned coil47 with `maxiter=20`. The latest schema-5
  run reached native/custom `12/19` and `13/23` iterations/evaluations.
  Initial and final objective differences were `2.7755575615628914e-17` and
  `2.7755575615628914e-17`; maximum final-parameter difference was
  `4.149377871680293e-09`; final gradient-infinity-norm difference was
  `2.889977063078508e-09`. Native/custom cold and warm times were
  `1.0344/1.0469 s` and `6.3525/0.06247 s`; solver RSS deltas were
  `0/345948 KiB`. Receipt:
  `.artifacts/custom-quasi-newton/20260801T-coil47-native-custom-cpu-parity-maxiter20-schema5/`.
  This is matched native/JAX CPU objective and endpoint evidence, not GPU or
  performance-promotion evidence.
- Green: an isolated `.venv-qn-gpu` with JAX 0.10.0 CUDA 12 support sees
  `cuda:0`; the recorded GPU run passed the optimizer trajectory/step tests
  and source-owned fixture checks under FP64 `jax_gpu_parity`. New schema-5 receipts record the
  JAX/XLA/SIMSOPT environment; the CPU environment remains separate.
- Green: source-owned coil47 custom L-BFGS on strict GPU, with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, converged in 12 iterations / 15
  evaluations. Final objective was `0.13786263284430206`; versus native CPU,
  the absolute objective difference was `2.7755575615628914e-17`, maximum
  parameter difference `4.143977844033031e-09`, and gradient-infinity-norm
  difference `2.8879265835549204e-09`. Cold/warm time was `22.1284/0.08087 s`
  and solver RSS delta `536252 KiB`. The GPU fast-intent run also converged to
  the same objective (`22.2542/0.08540 s`, `535620 KiB` delta). Receipts are
  under `.artifacts/custom-quasi-newton/20260801T-coil47-custom-gpu-*`.
- Diagnostic: Optax coil47 on strict GPU timed out at 120 seconds for 20 steps;
  its bounded two-step run took `11.7833/9.5193 s` cold/warm and reached
  objective `0.13786469652455852` (schema-5 receipt under
  `.artifacts/custom-quasi-newton/`). A default-preallocation run also hit
  CUDA allocation failures up to `23.52 GiB`; allocator settings are part of
  the GPU receipt contract, not hidden setup.
- Open transfer-guard issue: a global `JAX_TRANSFER_GUARD=disallow` rerun
  failed 32/46 optimizer tests during fixture setup or scalar assertions
  (`jnp.asarray` host literals and implicit boolean conversion), before solver
  execution. The normal strict-GPU lane passes; the dedicated boundary lane
  needs explicit device-placed test inputs and host-result assertions.
- Open: matched converged Boozer endpoint, compile/device-memory and StableHLO
  accounting, full compatibility suite, and tracked promotion receipts.
  Provider-child RSS isolation is green for the synthetic runner.
  Synthetic fixtures are not physics evidence. Four source-owned fixture tests
  marked `slow` pass in the required isolated CPU environment in 49.78 s
  combined. The GPU Boozer BFGS two-step child exceeded the 120-second
  watchdog; its initial objective/gradient parity remains covered by the
  source-owned fixture tests. The corrected CPU Boozer receipt reports 970,348 KiB
  native child RSS and 3,268,868 KiB custom child RSS. An earlier unisolated
  selector exceeded 180 seconds and about 17 GiB RSS; it is invalid diagnostic
  evidence, not a current fixture result. Runner receipts now time fixture
  construction separately from the solver, and child stdout is discarded so
  large JSON output cannot deadlock the watchdog pipe.
- Recorded checks (before the latest environment-schema test): 55 fast
  focused tests (48 core optimizer, trajectory, and result-schema tests plus
  seven runner/watchdog/measurement
  tests), plus four slow source-owned fixture tests
  passed. Direct BFGS edge
  and typed-callback checks, source-owned coil47/Boozer runner probes plus
  synthetic CPU BFGS/L-BFGS/Optax probes, the runner's Ruff check, formatting,
  compileall, and `git diff --check` pass. The targeted runtime/runner Ruff and
  formatting checks pass. The standalone typed step-runtime module passes
  Pyright with zero errors; the latest strict scoped project configuration
  reports 1,828 errors because its private
  optimizer dependencies and benchmark/test call sites are not yet typed. The
  project-wide Ruff command remains open under Ruff 0.16.1 (856 findings across
  the current source/test/benchmark tree); the touched Boozer source/test
  selector reports 84 findings. The isolated `.venv-qn-gpu` uses CUDA 12 on
  the local RTX 5090; the configured Landau A100 host remains unreachable
  (`No route to host`). The GPU lock is recorded in
  `benchmarks/environments/custom_quasi_newton_gpu.txt`. The GPU reruns add
  optimizer and source-owned fixture checks; their receipts are
  local working evidence only.
- Open diagnostic: a fresh-process `bfgs_quadratic` 47-variable pair produced
  no receipt; a native child remained around 300 MiB RSS for more than a minute
  before the probe was stopped. The same 47-variable custom call completed in
  about 0.56 s in an already-running process. Do not use the failed fresh
  process as a solver timing or parity result.
- Selected optimizer result/dispatch compatibility checks: 25 passed.
- The focused public compatibility selector (with `MPI4PY_RC_INITIALIZE=0`)
  passed 27 tests in 22.36 seconds; 133 unrelated tests were deselected.
- Public private-optimizer CPU checks: the Boozer private on-device selector
  passed 35 tests with one GPU-only closure test skipped; the broader public
  on-device selector passed 53 tests with seven GPU-only skips. Seven focused
  traceable exact/LM callable-cache tests passed after the cache fix. The full
  project compatibility suite remains open: the unbounded `traceable` selector
  was stopped at about 17.7 GiB RSS before completion.
- Green: all 13 legacy compatibility-shim tests pass when host-reference
  methods explicitly enter the `native_cpu` lane; on-device and traceable shim
  methods remain on the JAX lane. The prior RED was the validation process
  forcing `jax_cpu_parity` onto intentionally host-side legacy methods.
- Green: direct `target_minimize` coverage passed 12 tests; stage-two objective
  and dynamic-surface coverage passed 13 tests; and the planar, VMEC-free
  Boozer, and VMEC-hybrid example contract tests passed 16 tests on strict CPU.
  These are caller/contract checks, not matched native endpoint evidence.
- Green: the dedicated `_lbfgsb_scipy` transition/kernel compatibility suite
  passed 61/61 strict-CPU tests in 4m38s, including reverse-communication
  status ordering and result-schema coverage. End-to-end application parity is
  still a separate open gate.
- Green: strict-CPU end-to-end application parity passed for stage-two minimal
  (1/1), stage-two standard (1/1), and VMEC-free Boozer single-stage (2/2,
  including the executable parity-contract check). These tests compare native
  and JAX CPU inputs, construction fingerprints, initial observables, and
  bounded final outcomes; they are not GPU or performance receipts.
- Green: the remaining Stage-II parity cases passed on strict CPU: finite-build
  (2/2), planar coils (2/2), and stochastic (1/1). Across these mirrors,
  native/JAX CPU construction and bounded outcome contracts are green; strict
  GPU, wall-time, RSS, and device-memory qualification remain open.
- Probe: fixed-budget CPU `coil47` comparison (`maxiter=20`) reached the same
  final objective `0.137862632844302` for native, custom, and Optax. Native /
  custom / Optax iterations were `12 / 13 / 16`; cold seconds were
  `0.970634 / 5.692719 / 24.275724`; warm seconds were
  `0.766777 / 0.050979 / 23.552837`; child RSS was
  `460844 / 1010952 / 2375072 KiB`. Maximum final-parameter differences from
  native were `3.6456e-9` (custom) and `1.3207e-4` (Optax). This is dirty-tree
  exploratory CPU evidence, not the A100 promotion receipt.
- Probe: the same CPU `coil47` custom/Optax run in explicit `fast` intent reached
  the same final objective; custom / Optax cold seconds were `5.451687 /
  21.986368`, warm seconds `0.066188 / 21.502429`, and child RSS
  `989340 / 2171908 KiB`. This confirms the fast lane is executable, but is
  still dirty-tree CPU evidence without native or GPU comparison.
- Green: schema-4 transfer-audit rerun of the matched CPU `coil47` case kept
  the native/custom/Optax final objective at `0.137862632844302` (custom
  max-parameter difference from native `1.0232e-9`). The custom warm audit
  recorded `0` initialization, `26` advance, and `15` final-result transfer
  calls (`156` and `8,700` bytes for the latter two); native and Optax rows
  intentionally report empty custom-ledger fields. This remains CPU evidence
  on a dirty tree, not a GPU promotion receipt.
- Diagnostic: a fresh matched CPU Boozer run at `maxiter=100` did not close an
  endpoint parity gate. Native SciPy BFGS stopped after one failed inner-solve
  evaluation with `status=2` and objective `1000.0`; custom BFGS consumed all
  100 iterations with `status=1` and objective `8.469027533302147e-6`. Initial
  objective values still differed by only `2.168404344971009e-19`. The
  native failure penalty and the JAX rejected-objective policy are therefore a
  confirmed trajectory-semantics mismatch, not convergence evidence.

Review gate: not promotion-ready. The runner still needs matched converged
Boozer endpoint evidence, strict-GPU results, compile/device-memory and StableHLO
receipts, and a clean tracked manifest. The default eager L-BFGS path is the
existing three-kernel
design-B facade; the
generic accepted-step helper is not yet the production route. These are open
qualification items, not parity evidence. Historical RED revisions were not
preserved; current test files are post-hoc green evidence.

## Purpose

Refactor SIMSOPT's custom JAX BFGS and L-BFGS around the useful part of the
Optax structure: immutable state, fixed-shape compiled transitions, and a small
host driver for ordinary eager solves. SIMSOPT continues to own the algorithms,
SciPy-compatible behavior, and public results. Optax remains an explicit,
optional comparator; the custom solvers must not call Optax internals.

Optax's [official interface](https://github.com/google-deepmind/optax/blob/main/docs/getting_started.md)
is a composable `GradientTransformation` with `init`/`update`; its
[`scale_by_lbfgs`](https://github.com/google-deepmind/optax/blob/main/examples/lbfgs.ipynb)
is a gradient transform, not a SciPy-compatible L-BFGS-B
result/callback/status implementation. It therefore remains an explicit
comparator in this plan.

This plan supersedes the earlier proposal to remove custom BFGS/L-BFGS after
Optax qualification. The first implementation commit must update
`docs/jax_solver_algorithm_matrix.md` and
`docs/jax_solver_provider_coexistence_implementation_plan.md` accordingly. No
custom-provider removal or default-provider change is allowed until this plan's
compatibility, science, and performance gates pass.

## Goals

- [ ] Preserve `bfgs-ondevice` and `lbfgs-ondevice` method names, options,
      callbacks, statuses, counters, result fields, and SciPy parity behavior.
- [x] Use a fixed-shape accepted-step interface for normal eager solves, with
      total budgets outside step compilation.
- [x] Retain a whole-solve JAX route for callers that execute the optimizer
      under `jax.jit`; a Python host loop cannot consume traced optimizer state.
- [x] Keep line search on device with bounded JAX control flow.
- [ ] Allocate only current solver state and bounded history during normal
      execution; retain no full trajectory unless requested.
- [x] Keep JAX fast intent as the default after a JAX device is selected;
      parity remains explicit.
- [ ] Close through authentic RED -> GREEN -> REFACTOR tests and durable
      numerical, compile, timing, RSS, and device-memory receipts.

## Non-goals

- [ ] Do not replace the custom solvers with `optax.lbfgs`.
- [ ] Do not require Optax for custom BFGS/L-BFGS.
- [ ] Do not promise identical Optax, custom JAX, and SciPy trajectories.
- [ ] Do not add new constrained optimization behavior.
- [ ] Do not remove or deprecate the traced whole-solve route in this change.
- [ ] Do not hide objective compile or memory costs inside optimizer claims.
- [ ] Do not create a large benchmark matrix or fabricate historical RED
      revisions.
- [ ] Do not add an automatic dense-BFGS routing threshold or silently switch
      algorithms.

## Current facts and evidence limits

- `_bfgs.py` uses a fixed-shape eager step plus host driver for concrete inputs;
  the traced route still compiles the full solve with `lax.while_loop`, keeps
  staged `maxiter` in its cache identity, and uses the existing callback path.
- `_lbfgs.py` defaults to a host-observed driver over `start`, `search`, and
  `new_x_reentry` kernels. The eager dynamic-limit variants exclude
  `maxiter`/`maxfun` from their cache identity; the traced/static variants keep
  staged limits in their compilation contract.
- `BoozerSurfaceJAX.run_code_traceable()` and the traceable single-stage
  objective invoke whole-solve BFGS/L-BFGS inside a larger JIT. L-BFGS
  `monolithic_debug` is therefore a live compatibility route, not dead debug
  code.
- Before this refactor, L-BFGS history had shape `min(maxcor, maxiter)`. The
  eager path now allocates exactly `maxcor` slots; the old/new bytes and
  traced/static compatibility still require qualification below.
- BFGS inverse-Hessian state is `O(n^2)`. L-BFGS history is
  `O(n * maxcor)`, while the SciPy-compatible workspace also contains an
  `O(maxcor^2)` term (`2mn + 5n + 11m^2 + 8m` floating slots). Logical state,
  compiler/allocator amplification, and peak live bytes must be reported
  separately.
- `src/simsopt_jax/geo/optimizers/optimizer.py` owns legacy
  `target_minimize` callback/result conversion;
  `src/simsopt_jax/solve/dispatch.py` owns the typed solve API's callback,
  timing, and result normalization. `src/simsopt_jax/runtime/host_boundary.py`
  owns explicit device-to-host materialization. A private step runtime must not
  duplicate these owners.

The prior A100 observations—about 36 seconds for the custom 10-step probe,
about 12 seconds for the matched Optax probe, and roughly 34--36 GiB host RSS
during a stopped custom long run—were not archived with a durable raw result.
They are exploratory session notes, not a baseline or promotion evidence. Phase
0 must reproduce the failure from a clean revision before it influences design
or thresholds.

## Design

### Two execution paths, one algorithm

Each algorithm retains one mathematical transition implementation and exposes
two execution paths:

1. **Eager host-stepped path:** fixed-shape compiled transition plus a Python
   driver. This is the normal `bfgs-ondevice`/`lbfgs-ondevice` path when state is
   not traced.
2. **Traceable whole-solve path:** JAX control flow around the same transition
   primitives. This remains supported for `run_code_traceable()` and callers
   under `jax.jit`.

The route is selected from tracing context and the existing explicit
`lbfgs_run_mode` contract. No new public mode or compatibility flag is added.
Removal or deprecation of the whole-solve route requires a separate Tier-3
proposal with caller migration, warnings, a release timeline, and rollback.

### Ownership

Add `src/simsopt_jax/geo/optimizers/private/_step_runtime.py` with only:

```text
StepOps[
    StateT, TransitionT, ObservationT, HostObservationT, PayloadT, InitialT
] = immutable typed callable bundle
TransitionSink[HostObservationT, PayloadT] -> ContinueDecision
run_eager(ops, x0, limits, sink: TransitionSink[...]) -> StateT
BFGSStepOps: StepOps[
    _BFGSResults, _BFGSTransition, _BFGSObservation,
    _BFGSHostObservation, _BFGSResults, _BFGSResults
]
LBFGSBStepOps: typed facade over the specialized entry kernels
```

- `StepOps` has typed `initialize`, `advance`, `observe`,
  `host_observation`, and `payload` callables. The current BFGS eager path uses
  this generic driver. With physical design B, the current L-BFGS-B eager path
  remains a typed facade over the three specialized entry kernels rather than
  a merged branch graph; it is not yet routed through `run_eager`. The generic
  driver receives its bundle explicitly and performs no algorithm-tag dispatch.
- `TransitionSink` is supplied by the existing public owner. It receives the
  packed observation and, when requested, the typed payload, then returns the
  typed decision `CONTINUE` or `STOP`. A null sink requests no payload. The
  private driver never stores a callback in `StepOps` or in a compiled cache.
  `optimizer.py`/`dispatch.py` remain responsible for adapting callbacks and
  `StopIteration` into the sink decision.
- The BFGS state (`_BFGSResults`) and L-BFGS-B state
  (`_lbfgsb_scipy.LbfgsbState`) remain separate immutable pytrees.
- Algorithms own budget-transition timing, line search, status, and counters.
  The shared driver must not impose a generic zero-budget policy.
- Target eager contract: the no-observer path packs all stop/status/counter
  scalars into one fixed observation pytree and performs exactly one explicit
  `device_get` per eager `advance`, including terminal or nonaccepted
  transitions. Initialization and final packaging are counted separately. The
  observer path may add one typed callback/trace payload transfer containing
  `x`, `f`, gradient, counters, and status; its transfer count and bytes are
  recorded separately. Current BFGS and stepwise L-BFGS observer paths satisfy
  the packed-observation and phase-ledger contract; monolithic compatibility
  callbacks remain separately qualified.
- `geo/optimizers/optimizer.py` remains the compatibility owner for direct
  `target_minimize`; `solve/dispatch.py` remains the typed API owner. They keep
  callback conversion, `StopIteration`, timing, and public result
  normalization out of the private runtime.
- `runtime/host_boundary.py` remains the sole device-to-host boundary owner.
- State is per-run and immutable. Each objective keeps its own lock-protected
  LRU with capacity 8. The current implementation uses immutable tuple keys;
  the following is the target key contract and remains subject to the open
  collision tests:
  - common fields: algorithm, objective mode, dtype, flat parameter shape,
    structured cache token, pytree-adapter cache identity, closure-constant
    signature, and callback/trace compile policy;
  - BFGS fields: `value_and_grad`, norm, line-search bound, `gtol`, and `xrtol`;
  - L-BFGS-B fields: `maxcor`, `ftol`, `gtol`, `maxls`, bounds signature, and
    seeded-value/gradient compile policy.
  `maxiter` and `maxfun` are excluded only from the eager key because they are
  dynamic there. Collision tests vary every listed field. No global strong
  reference retains the objective or its closures. A per-key pending
  entry makes first compilation single-flight; failure removes the pending
  entry, and eviction never invalidates an executable already held by a solve.

### Physical compile boundary: decide by measurement

The public design requires one logical accepted-step operation. Two physical
implementations are allowed:

- **A:** one JIT containing reverse-communication branches; or
- **B:** a typed facade over the existing specialized entry kernels.

Phase 0 compares StableHLO size, compile time, peak RSS, and warm step time on a
cheap fixture and the coil objective. Choose the smaller implementation and
record the decision. Do not merge kernels merely for structural symmetry: a
unified branch graph can compile more code and consume more memory.

For the eager path, `maxiter` and `maxfun` are dynamic host/runtime limits.
The traceable whole-solve path may still compile for its staged loop bound.
`maxcor`, dtype, parameter shape, bounds shape, and objective closure structure
remain valid compilation identities because they change state shape or
generated code.

### Runtime modes and providers

These are orthogonal axes:

| Axis | Values | Meaning |
|---|---|---|
| Device | CPU, GPU | Selected JAX device |
| Execution intent | fast, parity | Existing runtime policy; fast is the JAX default |
| Provider | custom SIMSOPT, Optax, SciPy | Algorithm implementation; Optax is explicit, SciPy is the CPU oracle |

Optax is never a third execution intent and is never silently selected for a
custom method name.

## Implementation

### Phase 0 — provenance, root cause, and RED

- [ ] Work from an isolated worktree at a recorded commit. Record source
      status, Python/JAX/JAXLIB/SciPy/Optax versions, device, FP64 state,
      options, commands, exit codes, and fixture hashes.
- [x] Add the versioned runner
      `benchmarks/custom_quasi_newton_runtime.py` for deterministic quadratic,
      Rosenbrock, coil47, and Boozer fixtures, with synchronization,
      child-process measurements, and a JSON schema. The physics cases remain
      endpoint/promotion evidence only after matched native/C++ endpoint
      contracts close; coil47 and Boozer now expose matched SIMSOPT-native
      objective callbacks for initial-state checks.
- [x] Add `benchmarks/fixtures/custom_quasi_newton.py` and
      `benchmarks/fixtures/custom_quasi_newton_cases.json` as the current
      runtime-contract fixture SSOT. The source-owned physics builders are
      present; full endpoint and device qualification remain open:
  - [x] `coil47` is a deterministic, VMEC-free coil preoptimization slice
        derived from the curve/current construction in
        `examples/jax/3_Advanced/single_stage_optimization.py`; its frozen
        analytic surface keeps the fixture independent of VMEC and mutable
        files. The generator asserts 47 free variables and records geometry,
        quadrature, current-coordinate, objective, FP64, and seed metadata.
        Its certificate includes matched native/JAX objective parity on CPU;
        the custom GPU endpoint is now recorded; Optax and full performance
        qualification remain open.
  - [x] `boozer` is the deterministic vacuum case derived from
        `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py`;
        it records surface/field construction, exercises the compiled
        `run_code_traceable()` route, and exposes a matched native
        `BoozerSurface` objective callback. The fixture selects
        `optimizer_backend="ondevice"` explicitly. The host-controlled eager
        `run_code()` route is intentionally not invoked by this strict fixture:
        strict JAX rejects that fallback and its compatibility tests remain a
        separate gate.
  - [x] The runner JSON records generator/source hashes, full initial vectors,
        solver option vectors, expected initial observables, final certificate
        fields, and predeclared tolerances. Neither fixture may read VMEC, a
        network path, or mutable user data.
  - [x] Add the focused runtime and SciPy trajectory tests:
  - [x] `tests/jax/solve/test_custom_quasi_newton_step_runtime.py` (42 tests;
        40 fast contracts and two `slow` source-owned physics probes);
  - [x] `tests/jax/solve/test_lbfgsb_trajectory_parity.py` (accepted-step,
        frozen FP64 fields, deferred-`maxfun`, and observer-equivalence tests).
        These are current-worktree evidence until committed;
        they are not tracked promotion receipts.
  - [x] `tests/benchmarks/test_custom_quasi_newton_runtime.py` (10
        provider-child, fixture-cost, provenance, native-objective, and
        memory-contract tests).
  - [x] focused additions/updates to
        `tests/geo/test_boozersurface_jax_private.py` and
        `tests/geo/test_boozersurface_jax.py`, including the traceable closure
        cache and signature-contract regressions.
- [x] Freeze accepted-step behavior for FP64 quadratic and Rosenbrock cases:
      `x`, objective, gradient, step length, line-search task/status,
      iterations, `nfev`, and `njev` are pinned in the focused BFGS/L-BFGS
      tests. These are tolerance-based frozen contracts, not GPU bitwise claims.
- [x] Add compile-count regression tests: after one warm solve, changing only
      `maxiter` or `maxfun` creates no new eager step executable. The tests
      observe compilation behavior without pinning private cache-key spelling.
- [x] Pin separate zero-budget semantics: BFGS `maxiter=0` takes no step;
      L-BFGS-B preserves SciPy's deferred stop check and accepts one `NEW_X`
      before reporting the limit. `test_zero_budget_preserves_bfgs_and_lbfgs_limit_timing`
      covers both contracts.
- [ ] Run the five-cell root-cause matrix: cheap versus coil objective,
      `maxcor=10` versus `300`, short versus long budget, and compile-only
      versus warm execution. Record iteration progress, compile events,
      per-step timing, objective-only timing, an RSS time series, StableHLO
      size, and executable count. Do not infer a compile defect merely from a
      long elapsed time.
      Use `benchmarks/lbfgs_ondevice_compile_shape.py`; its default diagnostic
      excludes the legacy generic/monolithic kernels. Add
      `--include-legacy-kernels` only under an externally watched process.
- [ ] Compare physical compile designs A and B. If the objective graph—not the
      optimizer control/state—is the dominant cost, stop this refactor claim
      and open an objective-specific plan.
- [x] Bound each diagnostic in an exact child process. The runner sends TERM,
      then KILL after a grace period, at 120 seconds or 8 GiB RSS, and tracks
      the child PID directly; no broad process-name matching is used. The
      timeout and RSS branches have bounded contract tests.

### Phase 1 — shared eager driver

- [x] Implement the typed private protocol without `Any`, dynamic imports,
      mutable dictionaries, runtime algorithm tags, or public-result logic.
- [x] Keep runtime limits immutable host data. Where L-BFGS-B needs limits in a
      compiled transition, pass total limits dynamically and preserve SciPy's
      post-`NEW_X` checks. Never truncate an in-progress line search to prevent
      `maxfun` overshoot.
- [x] Implement separate scalar-only and callback/trace observation paths.
- [x] Complete the transfer audit: assert one packed scalar transfer per eager
      `advance`, including terminal and nonaccepted transitions, and report
      separate initialization, callback/trace, and final-result transfers.
      The context-local host-boundary ledger is emitted in runner schema 5;
      tests cover full eager L-BFGS transitions with and without callbacks,
      terminal/nonaccepted packets, and BFGS observer-phase attribution.
- [x] Cover initial convergence, zero budgets, evaluation exhaustion,
      nonfinite state, callback `StopIteration`, status mapping, and concurrent
      independent solves with no shared mutable state or crossed callbacks.
      Initial-convergence, zero-budget, and L-BFGS-B evaluation-exhaustion
      behavior are covered for both public and private paths; two-thread BFGS
      and L-BFGS isolation tests cover independent callback ownership.
- [x] Prove the null sink and every non-stopping observer produce the same
      trajectory. For a stopping observer, prove the accepted prefix, callback
      order, status, counters, and stop point match the frozen contract.
      Current tests cover non-stopping BFGS/L-BFGS-B and multi-step callback
      order, accepted-prefix stopping, and concurrent callback ownership.

### Phase 2 — BFGS

- [x] Extract immutable initialization and one direction/line-search/curvature
      transition from `_minimize_bfgs_private`.
- [x] Preserve strong-Wolfe constants, lower-precision decreasing fallback,
      curvature validation, `gtol`, `xrtol`, line-search status, counters, and
      final re-evaluation; the focused private compatibility selectors and
      frozen contracts cover these fields.
- [x] Route ordinary eager calls through the host driver; retain the traceable
      whole-solve route using the same transition primitives.
- [x] Report dense inverse-Hessian logical bytes and a conservative derived
      peak-live upper bound during the update, including simultaneous old/new
      Hessians and intermediates. Record the no-donation policy. The runner
      emits this accounting without adding a routing threshold.
- [ ] Measure actual dense-BFGS peak live bytes during the update on supported
      CPU/GPU devices and compare them with the derived upper bound.

### Phase 3 — L-BFGS-B

- [x] Preserve `_lbfgsb_scipy.py` transition equations, reverse-communication
      task/status ordering, callback order, counters, `ftol`, `gtol`, `maxls`,
      seeded value/gradient, and inverse-Hessian extraction.
- [x] Implement the logical accepted-step operation using the Phase-0-selected
      physical compile design.
- [x] Remove `maxiter`/`maxfun` from eager init/step/result static closures and
      cache identities without changing algorithm-owned stop timing.
- [x] Allocate exactly `maxcor` history slots, independent of `maxiter`; update
      the pinned state-shape test and report the old/new byte difference for
      `maxiter < maxcor`.
- [x] Retain no accepted iterates normally. Keep explicit trace capture within
      its existing byte cap; the no-trace and oversized-trace tests cover both
      paths.
- [x] Preserve `monolithic_debug` and traced `run_code_traceable()` behavior.
- [x] Replace the objective-attached unbounded compiled-wrapper dictionary with
      the per-objective capacity-8 LRU and single-flight admission specified
      above.
- [x] Complete basic cache qualification: test key identity, owner garbage
      collection, eviction while another solve holds an entry, failed-
      compilation cleanup, cardinality under shape/policy churn, and
      concurrent single-flight first use. Structured-adapter collision coverage
      now has a tree-definition key-separation test; full adapter execution
      collision coverage remains part of Phase 4.

### Phase 4 — public compatibility and application tests

- [ ] Keep public method names and provider routing unchanged.
- [ ] Inventory all callers of `lbfgs_run_mode`, `maxcor`, `maxfun`, callbacks,
      seeded gradients, `hess_inv`, status, and counters.
- [ ] Complete direct `target_minimize`, BoozerSurface eager and traceable
      paths, stage-two, and single-stage caller coverage. Contract smoke
      selectors are green; full eager/traceable and application-scale closure
      remains open.
- [x] Legacy `jax_minimize`/`jax_least_squares` shim routes are covered for
      host-reference, on-device, trace, and Optax/Optimistix methods; host
      reference tests select `native_cpu` explicitly.
- [x] Direct target, stage-two objective, and single-stage example contract
      selectors pass in the isolated strict CPU lane; full application-scale
      endpoint parity remains open.
- [ ] Compare every accepted state against the pre-refactor custom solver.
      Compare L-BFGS-B against pinned SciPy with matched options. Separate
      bitwise, tolerance, and equivalent-endpoint verdicts.
- [ ] Verify non-stopping callback/no-callback and trace/no-trace configurations
      do not change numerical results. Verify `StopIteration` produces the
      intended frozen trajectory prefix and terminal result.
- [ ] Verify eager transition/math kernels and no-observer paths import or
      execute no SciPy, Optax, Optimistix, NumPy numerical work, or host
      callback. For supported callbacks inside the traced whole-solve route,
      retain `jax.debug.callback` only at the accepted-step observation
      boundary and freeze its current ordering flag, payload, callback order,
      status, and counter behavior.

### Phase 5 — lean physics and performance qualification

- [ ] Qualify L-BFGS on the runner's 47-parameter coil case and BFGS on its
      representative Boozer case using native CPU, JAX CPU, and strict JAX GPU.
- [x] Strict-CPU application parity smoke is green for stage-two minimal,
      stage-two standard, and VMEC-free Boozer single-stage; the custom GPU
      runner lane for the 47-parameter coil is also green, while the combined
      Boozer/performance qualification remains open.
- [x] Strict-CPU application parity is also green for finite-build, planar,
      and stochastic Stage-II mirrors; these are outcome-contract evidence,
      not performance receipts.
- [ ] Compare initial and final objective components, parameters, invariant
      geometry observables, gradient infinity norm, constraints, iterations,
      evaluations, raw status, and stopping reason.
- [ ] Label capped, converged, failed, and callback-stopped states directly.
      A finite decrease or lower objective alone is not convergence.
- [ ] Measure cold compile, warm optimizer, total wall time, peak RSS, peak
      device memory, StableHLO size, and executable count. Synchronize timed
      boundaries and exclude export/plotting.
- [ ] Compare Optax and custom overhead at a fixed accepted-step budget, then
      compare time to the same scientific certificate. Report Optax line-search
      evaluations as unavailable unless the runner measures them; do not infer
      `nfev` from outer iterations.
- [x] A fixed-budget CPU `coil47` Optax/custom/native comparison is recorded
      above; strict-GPU/A100 repetition and certificate-time comparison remain
      open.
- [ ] For the 47-parameter A100 case, require:
  - [ ] no 120-second or 8-GiB guard trip;
  - [ ] warm custom time no more than `2x` matched Optax warm time;
  - [ ] custom peak RSS no more than `1.5x` matched Optax peak RSS;
  - [ ] zero new eager step compilations when only budgets change; and
  - [ ] no monotonic RSS/device-memory or executable-count growth across five
        warm repeats.
- [ ] These `2x` and `1.5x` thresholds are predeclared initial promotion gates.
      Change them only by reviewing this plan before GREEN data are collected.
      The work need not prove every small example is faster than native CPU.

### Phase 6 — refactor, rollout, and rollback

- [ ] Remove duplicated eager host-loop mechanics only after BFGS and L-BFGS
      pass independently. Keep all mathematics algorithm-owned.
- [ ] Keep the runtime/transition implementation commits separate from one
      routing-only commit so rollback does not discard validated primitives or
      tests.
- [ ] Expand `[tool.pyright].include` and add
      `pyright.custom-quasi-newton.json`, scoped to the changed private
      optimizer, fixture, runner, and test paths with
      `typeCheckingMode: "strict"`. Do not add blanket ignores or permit
      unknown/`Any` types in the new protocol.
- [ ] Update the solver matrix, provider plan, public docs, and examples to use
      the device/intent/provider taxonomy above.
- [ ] Record the pre-routing commit as the rollback point. If science,
      callbacks, traced execution, compilation, or memory regresses, revert the
      routing commit to the current host-observed/whole-solve paths. Solver state
      is process-local, so no persisted-state migration is required.
- [ ] Rehearse rollback in a clean worktree: revert only the isolated routing
      commit, run the frozen optimizer and eager/traceable Boozer compatibility
      selectors, and archive the reverted commit, commands, exit codes, and
      checksums in the receipt. Then return to the candidate commit and rerun
      the same selectors.
- [ ] Do not remove the traceable whole-solve implementation in this plan.

## Receipt contract

The runner writes local working data to
`.artifacts/custom-quasi-newton/<run-id>/`. Promotion additionally requires:

- [ ] a tracked manifest at
      `docs/receipts/custom-quasi-newton/<run-id>/manifest.json` with schema
      version, commit, clean status, environment lock hashes, device, commands,
      exit codes, artifact checksums, verdicts, and archive URI;
- [ ] tracked compact `metrics.json` and `summary.md` beside the manifest; and
- [ ] raw logs/JSON copied to the archive URI and verified from a fresh process.

The ignored local `.artifacts/` copy alone is never authority. A result is
incomplete if the archive is absent, a checksum differs, the environment is an
unsupported overlay, or a child is timed out/killed.

## Validation

### Supported environments

Create clean Python 3.11+ environments from the candidate checkout:

```bash
uv venv --python 3.11 .venv-qn-cpu
uv pip install --python .venv-qn-cpu/bin/python \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -e ".[JAX,dev,ALGS]"
uv pip freeze --python .venv-qn-cpu/bin/python > benchmarks/environments/custom_quasi_newton_cpu.txt

uv venv --python 3.11 .venv-qn-gpu
uv pip install --python .venv-qn-gpu/bin/python \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -e ".[JAX_GPU,dev,ALGS]"
uv pip freeze --python .venv-qn-gpu/bin/python > benchmarks/environments/custom_quasi_newton_gpu.txt
```

Phase 0 creates and reviews the constraints file with exact SciPy and Optax
versions compatible with the project-pinned JAX/JAXLIB `0.10.0`; the initial
Optax comparator pin is `0.2.8`. Record the resolved transitive environment.
Do not use the system Python or an overlay from another checkout.

For non-MPI optimizer tests, set `MPI4PY_RC_INITIALIZE=0` in the child
environment so importing `simsopt.field` does not auto-initialize MPI or open a
host X11 session. Do not apply that setting to a VMEC/MPI workflow that
explicitly owns MPI initialization.

### CPU

```bash
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  tests/jax/solve/test_optimizer_result_schema.py \
  tests/geo/test_lbfgsb_scipy_jax_kernels.py \
  -m "not slow"

# Source-owned fixture construction is intentionally a separate slow lane.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/benchmarks/test_custom_quasi_newton_runtime.py -m slow

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_boozersurface_jax.py \
  tests/jax/solve/test_driver_dispatch.py \
  tests/jax/solve/test_compat_shim_translation.py \
  tests/jax/examples/test_single_stage_vmec_hybrid_example.py \
  -k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"
```

Each CPU process must assert `jax.default_backend() == "cpu"` and record
`jax.devices()` before tests or probes.

Run the matched native/custom CPU cases through the versioned runner. The
source-owned physics fixture builders are separate slow probes; the runner's
solver timings begin after fixture construction.

The current executable smoke cases are `--cases rosenbrock`, `--cases coil47`,
and `--cases boozer` (or `bfgs_quadratic` with `--providers native,custom` for
dense BFGS). Coil and Boozer use different solver methods, so they run as
separate commands. Coil47 and Boozer now have matched native/JAX CPU
initial-objective paths; endpoint and GPU qualification remain separate gates.

```bash
RUN_DIR=".artifacts/custom-quasi-newton/$(date -u +%Y%m%dT%H%M%SZ)-cpu-smoke"
mkdir -p "$RUN_DIR"

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases rosenbrock --output "$RUN_DIR"

# Source-owned coil physics smoke; native/JAX CPU objective parity is covered.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases coil47 --output "$RUN_DIR"

# Source-owned Boozer physics smoke; native and custom providers share the
# initial objective/gradient contract. Capped endpoints are diagnostic only.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases boozer --output "$RUN_DIR"
```

### Strict GPU

Run the fast focused tests and the physics cases in fresh CUDA-only processes.
The `slow` fixture-construction tests are a separate qualification step; do
not include them in the fast contract command:

```bash
RUN_ROOT=".artifacts/custom-quasi-newton/$(date -u +%Y%m%dT%H%M%SZ)-gpu-parity"
CUSTOM_RUN_DIR="$RUN_ROOT/custom"
OPTAX_RUN_DIR="$RUN_ROOT/optax"
mkdir -p "$CUSTOM_RUN_DIR" "$OPTAX_RUN_DIR"

MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  -m "not slow"

# Broader Boozer compatibility selectors are a separate bounded qualification
# run; do not fold their fixture construction into the fast contract lane.
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_boozersurface_jax_private.py \
  -k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"

# Slow source-owned fixture construction is run separately when GPU resources
# are available; it is not part of the fast contract result.
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/benchmarks/test_custom_quasi_newton_runtime.py -m slow

JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers custom \
  --cases coil47 --output "$CUSTOM_RUN_DIR"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers custom \
  --cases boozer --output "$CUSTOM_RUN_DIR"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers optax \
  --cases coil47 --output "$OPTAX_RUN_DIR"
```

The runner must fail before execution unless the requested CPU lane reports
`jax.default_backend() == "cpu"`, or the requested GPU lane reports a CUDA/
ROCm/GPU backend and every visible JAX device is a GPU. Every leaf of inputs,
closure constants, initialized solver state, step output, and pre-host final
result is placed on a GPU. Use
`XLA_PYTHON_CLIENT_PREALLOCATE=false` with
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` for matched GPU comparisons and record
both settings in the schema-5 receipt. Transfer-guard checks are a separate
boundary lane: do not enable a global disallow guard for this fixture suite,
whose setup intentionally constructs device inputs from host literals.

### Quality and closure

```bash
.venv-qn-cpu/bin/pyright --project pyright.custom-quasi-newton.json --warnings
.venv-qn-cpu/bin/ruff check src/simsopt_jax tests/jax tests/geo benchmarks
.venv-qn-cpu/bin/ruff format --check src/simsopt_jax tests/jax tests/geo benchmarks
.venv-qn-cpu/bin/python -m compileall -q src/simsopt_jax tests/jax benchmarks
```

Run the full project suite after focused validation. Final closure must be
replayed from a clean detached worktree at the candidate commit; run
`git diff --check` there, then verify the tracked manifest against the external
archive. This prevents unrelated dirty-tree files from entering the verdict.

## Completion criteria

- [ ] Eager custom BFGS/L-BFGS use the fixed-shape step runtime; traced callers
      retain a supported whole-solve JAX path.
- [ ] Changing only `maxiter`/`maxfun` causes no new eager init/step executable,
      and the `maxcor` allocation change is measured and documented.
- [ ] Public options, callbacks, statuses, counters, result fields, zero-budget
      semantics, and L-BFGS inverse-Hessian behavior pass frozen tests.
- [ ] Native SciPy/SIMSOPT remains the parity oracle; Optax remains explicit.
- [ ] CPU and strict-GPU tests pass in supported isolated environments without
      weakened tolerances or CPU fallback.
- [ ] Physics cases meet the predeclared gates with durable raw evidence.
- [ ] Normal execution retains no trajectory and uses only audited host
      boundaries.
- [ ] Focused and broad tests, Pyright, Ruff, formatting, compileall, and clean
      `git diff --check` pass.
- [ ] Solver architecture docs agree with this plan, and rollback is proven.

## Resolved decisions

- Keep the whole-solve route for traced production callers.
- Use one packed scalar host observation per eager transition; defer fixed-size
  multi-step chunks until this path is correct and measured.
- Report dense-BFGS memory; do not add automatic routing.
- Compare Optax in the supported project JAX/JAXLIB environment and record the
  exact resolved Optax version.
- Use the `2x` warm-time and `1.5x` RSS gates above unless this plan is revised
  before implementation.
