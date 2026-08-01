# Custom JAX BFGS/L-BFGS Step Runtime Implementation Plan

**Status:** Proposed

**Last updated:** 2026-08-01

**Change tier:** Tier 3 — public solver behavior and traced call paths

## Purpose

Refactor SIMSOPT's custom JAX BFGS and L-BFGS around the useful part of the
Optax structure: immutable state, fixed-shape compiled transitions, and a small
host driver for ordinary eager solves. SIMSOPT continues to own the algorithms,
SciPy-compatible behavior, and public results. Optax remains an explicit,
optional comparator; the custom solvers must not call Optax internals.

This plan supersedes the earlier proposal to remove custom BFGS/L-BFGS after
Optax qualification. The first implementation commit must update
`docs/jax_solver_algorithm_matrix.md` and
`docs/jax_solver_provider_coexistence_implementation_plan.md` accordingly. No
custom-provider removal or default-provider change is allowed until this plan's
compatibility, science, and performance gates pass.

## Goals

- [ ] Preserve `bfgs-ondevice` and `lbfgs-ondevice` method names, options,
      callbacks, statuses, counters, result fields, and SciPy parity behavior.
- [ ] Use a fixed-shape accepted-step interface for normal eager solves, with
      total budgets outside step compilation.
- [ ] Retain a whole-solve JAX route for callers that execute the optimizer
      under `jax.jit`; a Python host loop cannot consume traced optimizer state.
- [ ] Keep line search on device with bounded JAX control flow.
- [ ] Allocate only current solver state and bounded history during normal
      execution; retain no full trajectory unless requested.
- [ ] Keep JAX fast intent as the default after a JAX device is selected;
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

- `_bfgs.py` compiles the full solve with `lax.while_loop`; its cache identity
  includes `maxiter` and callbacks run through `jax.debug.callback`.
- `_lbfgs.py` defaults to a host-observed driver over `start`, `search`, and
  `new_x_reentry` kernels. Its caches specialize on `maxiter`/`maxfun`.
- `BoozerSurfaceJAX.run_code_traceable()` and the traceable single-stage
  objective invoke whole-solve BFGS/L-BFGS inside a larger JIT. L-BFGS
  `monolithic_debug` is therefore a live compatibility route, not dead debug
  code.
- L-BFGS history currently has shape `min(maxcor, maxiter)`. Removing `maxiter`
  from compilation requires allocating `maxcor` slots consistently; that is a
  deliberate memory and diagnostic change covered below.
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
StepOps[StateT, TransitionT, PayloadT] = immutable typed callable bundle
TransitionSink[ObservationT, PayloadT] -> ContinueDecision
run_eager(ops, x0, limits, sink: TransitionSink[...]) -> StateT
BFGSStepOps: StepOps[BFGSState, BFGSTransition, BFGSPayload]
LBFGSBStepOps: StepOps[LBFGSBState, LBFGSBTransition, LBFGSBPayload]
```

- `StepOps` has typed `initialize`, `advance`, `pack_observation`, and
  `callback_payload` callables. Concrete builders create the BFGS and L-BFGS-B
  bundles; the generic driver receives the bundle explicitly and performs no
  algorithm-tag dispatch.
- `TransitionSink` is supplied by the existing public owner. It receives the
  packed observation and, when requested, the typed payload, then returns the
  typed decision `CONTINUE` or `STOP`. A null sink requests no payload. The
  private driver never stores a callback in `StepOps` or in a compiled cache.
  `optimizer.py`/`dispatch.py` remain responsible for adapting callbacks and
  `StopIteration` into the sink decision.
- `BFGSState` and `LBFGSBState` remain separate immutable pytrees.
- Algorithms own budget-transition timing, line search, status, and counters.
  The shared driver must not impose a generic zero-budget policy.
- The no-observer path packs all stop/status/counter scalars into one fixed
  observation pytree and performs exactly one explicit `device_get` per eager
  `advance`, including terminal or nonaccepted transitions. Initialization and
  final packaging are counted separately. The observer path may add one typed
  callback/trace payload transfer containing `x`, `f`, gradient, counters, and
  status; its transfer count and bytes are recorded separately.
- `geo/optimizers/optimizer.py` remains the compatibility owner for direct
  `target_minimize`; `solve/dispatch.py` remains the typed API owner. They keep
  callback conversion, `StopIteration`, timing, and public result
  normalization out of the private runtime.
- `runtime/host_boundary.py` remains the sole device-to-host boundary owner.
- State is per-run and immutable. Each objective keeps its own lock-protected
  LRU with capacity 8. Algorithm-specific immutable key types contain:
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
- [ ] Add the versioned runner
      `benchmarks/custom_quasi_newton_runtime.py`. It owns the quadratic,
      Rosenbrock, coil, and Boozer invocation; synchronization; and JSON schema.
- [ ] Add `benchmarks/fixtures/custom_quasi_newton.py` and
      `benchmarks/fixtures/custom_quasi_newton_cases.json` as the fixture SSOT:
  - [ ] `coil47` is the deterministic, VMEC-free coil-preoptimization slice
        derived from `examples/jax/3_Advanced/single_stage_optimization.py`;
        the generator asserts 47 free variables and records every geometry,
        quadrature, current-coordinate, weight, solver-option, FP64, and seed
        value.
  - [ ] `boozer` is the deterministic vacuum case derived from
        `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py`;
        it records surface/field construction and exercises both eager
        `run_code()` and compiled `run_code_traceable()` routes.
  - [ ] The JSON records generator/source hashes, full initial vectors, option
        vectors, expected initial observables, final certificate fields, and
        predeclared tolerances. Neither fixture may read VMEC, a network path,
        or mutable user data.
- [ ] Add planned tests:
  - [ ] `tests/jax/solve/test_custom_quasi_newton_step_runtime.py`;
  - [ ] `tests/jax/solve/test_lbfgsb_trajectory_parity.py` (currently untracked,
        so it is not baseline evidence until reviewed and committed);
  - [ ] focused additions to `tests/geo/test_boozersurface_jax_private.py` and
        `tests/geo/test_boozersurface_jax.py`.
- [ ] Freeze accepted-step behavior for FP64 quadratic and Rosenbrock cases:
      `x`, objective, gradient, step length, line-search task/status,
      iterations, `nfev`, and `njev`.
- [ ] Add RED observable compile-count tests: after one warm solve, changing
      only `maxiter` or `maxfun` must create no new eager step executable. Do
      not pin private cache-key spelling.
- [ ] Pin separate zero-budget semantics: BFGS `maxiter=0` takes no step;
      L-BFGS-B preserves SciPy's deferred stop check and may accept one
      `NEW_X` before reporting the limit.
- [ ] Run a four-probe root-cause matrix: cheap versus coil objective,
      `maxcor=10` versus `300`, short versus long budget, and compile-only
      versus warm execution. Record iteration progress, compile events,
      per-step timing, objective-only timing, an RSS time series, StableHLO
      size, and executable count. Do not infer a compile defect merely from a
      long elapsed time.
- [ ] Compare physical compile designs A and B. If the objective graph—not the
      optimizer control/state—is the dominant cost, stop this refactor claim
      and open an objective-specific plan.
- [ ] Bound each diagnostic in an exact child process. A watchdog sends TERM,
      then KILL after a grace period, at 120 seconds or 8 GiB RSS. Track the
      child PID directly; do not use broad process-name matching.

### Phase 1 — shared eager driver

- [ ] Implement the typed private protocol without `Any`, dynamic imports,
      mutable dictionaries, runtime algorithm tags, or public-result logic.
- [ ] Keep runtime limits immutable host data. Where L-BFGS-B needs limits in a
      compiled transition, pass total limits dynamically and preserve SciPy's
      post-`NEW_X` checks. Never truncate an in-progress line search to prevent
      `maxfun` overshoot.
- [ ] Implement separate scalar-only and callback/trace observation paths.
- [ ] Assert one packed scalar transfer per eager `advance`, including terminal
      and nonaccepted transitions. Assert and report separate initialization,
      callback/trace, and final-result transfers.
- [ ] Cover initial convergence, zero budgets, evaluation exhaustion,
      nonfinite state, callback `StopIteration`, status mapping, and concurrent
      independent solves with no shared mutable state or crossed callbacks.
- [ ] Prove the null sink and every non-stopping observer produce the same
      trajectory. For a stopping observer, prove the accepted prefix, callback
      order, status, counters, and stop point match the frozen contract.

### Phase 2 — BFGS

- [ ] Extract immutable initialization and one direction/line-search/curvature
      transition from `_minimize_bfgs_private`.
- [ ] Preserve strong-Wolfe constants, lower-precision decreasing fallback,
      curvature validation, `gtol`, `xrtol`, line-search status, counters, and
      final re-evaluation.
- [ ] Route ordinary eager calls through the host driver; retain the traceable
      whole-solve route using the same transition primitives.
- [ ] Report dense inverse-Hessian logical bytes and derived/measured peak live
      bytes during the update, including simultaneous old/new Hessians and
      intermediates. Record the no-donation policy. Do not reject, reroute, or
      add a public threshold in this change.

### Phase 3 — L-BFGS-B

- [ ] Preserve `_lbfgsb_scipy.py` transition equations, reverse-communication
      task/status ordering, callback order, counters, `ftol`, `gtol`, `maxls`,
      seeded value/gradient, and inverse-Hessian extraction.
- [ ] Implement the logical accepted-step operation using the Phase-0-selected
      physical compile design.
- [ ] Remove `maxiter`/`maxfun` from eager init/step/result static closures and
      cache identities without changing algorithm-owned stop timing.
- [ ] Allocate exactly `maxcor` history slots, independent of `maxiter`; update
      the pinned state-shape test and report the old/new byte difference for
      `maxiter < maxcor`.
- [ ] Retain no accepted iterates normally. Keep explicit trace capture within
      its existing byte cap.
- [ ] Preserve `monolithic_debug` and traced `run_code_traceable()` behavior.
- [ ] Replace the objective-attached unbounded compiled-wrapper dictionary with
      the per-objective capacity-8 LRU specified above. Test key identity,
      owner garbage collection, eviction while another solve holds an entry,
      failed-compilation cleanup, cardinality under shape/policy churn, and
      concurrent single-flight first use.

### Phase 4 — public compatibility and application tests

- [ ] Keep public method names and provider routing unchanged.
- [ ] Inventory all callers of `lbfgs_run_mode`, `maxcor`, `maxfun`, callbacks,
      seeded gradients, `hess_inv`, status, and counters.
- [ ] Test direct `target_minimize`, BoozerSurface eager and traceable paths,
      stage-two, and single-stage callers.
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
python3.11 -m venv .venv-qn-cpu
.venv-qn-cpu/bin/python -m pip install \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -e ".[JAX,dev,ALGS]"
.venv-qn-cpu/bin/python -m pip freeze > benchmarks/environments/custom_quasi_newton_cpu.txt

python3.11 -m venv .venv-qn-gpu
.venv-qn-gpu/bin/python -m pip install \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -e ".[JAX_GPU,dev,ALGS]"
.venv-qn-gpu/bin/python -m pip freeze > benchmarks/environments/custom_quasi_newton_gpu.txt
```

Phase 0 creates and reviews the constraints file with exact SciPy and Optax
versions compatible with the project-pinned JAX/JAXLIB `0.10.0`; the initial
Optax comparator pin is `0.2.8`. Record the resolved transitive environment.
Do not use the system Python or an overlay from another checkout.

### CPU

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  tests/jax/solve/test_optimizer_result_schema.py \
  tests/geo/test_lbfgsb_scipy_jax_kernels.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src \
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

Run the matched native/custom CPU cases through the versioned runner:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases coil47,boozer --output "$RUN_DIR"
```

### Strict GPU

Run the same focused tests and both physics cases in a fresh CUDA-only process:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_TRANSFER_GUARD=disallow \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_boozersurface_jax.py \
  -k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_TRANSFER_GUARD=disallow \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers custom \
  --cases coil47,boozer --output "$RUN_DIR"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_TRANSFER_GUARD=disallow \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers optax \
  --cases coil47 --output "$RUN_DIR"
```

The runner must fail before execution unless `jax.default_backend() == "gpu"`,
every visible JAX device is a GPU, and every leaf of inputs, closure constants,
initialized solver state, step output, and pre-host final result is placed on a
GPU. `JAX_TRANSFER_GUARD=disallow` permits the explicit
scalar/callback/final-result boundaries owned above; structural tests must
reject hidden callbacks or NumPy computation. Use the same allocator setting
for matched GPU comparisons and record it.

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
