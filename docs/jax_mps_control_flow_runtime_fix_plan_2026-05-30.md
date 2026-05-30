# JAX-MPS Control-Flow Runtime Fix Implementation Plan

## Purpose

This plan defines a measured path for upgrading or customizing `jax-mps` so the
SIMSOPT `scipy-jax` lane can use Apple Silicon MPS without spending most of its
time in host-synchronized StableHLO control flow. The plan is intentionally
runtime-first: the current evidence points at `jax-mps`/MLX execution semantics,
not at SciPy L-BFGS-B or the physics objective alone.

## Goals

- Prove or falsify that `stablehlo.while` host synchronization dominates the
  MPS slowdown for the SIMSOPT 3-iteration smoke.
- Add enough `jax-mps` instrumentation to separate compile time, dispatch time,
  loop trip count, `mlx::core::eval`, scalar condition reads, and body execution.
- Prototype a semantics-preserving control-flow improvement for loops that can
  be proven safe, with strict fallback to current behavior when the proof fails.
- Keep the SIMSOPT MPS lane smoke-only until correctness and performance gates
  pass against CPU and CUDA reference artifacts.
- Produce an upstreamable patch split if the generic `jax-mps` fix works, or a
  local custom-kernel decision record if generic control flow is blocked by MLX
  API constraints.

## Non-Goals

- Do not attempt to make `jax-mps` feature-equivalent to XLA-CUDA in one pass.
- Do not claim production SIMSOPT support on MPS while the repo's
  `jax_mps_smoke` policy and the installed `jax-mps` lane exclude float64 for
  this workload.
- Do not change SciPy L-BFGS-B semantics or move the optimizer itself onto MPS.
- Do not promote MPS as a default production backend in `simsopt-jax`.
- Do not stop or alter unrelated long-running SIMSOPT or autoresearch processes.

## Current Context

### Confirmed Source Facts

- Local `jax-mps` source: `/Users/suhjungdae/code/opensource/jax-mps`, clean at
  `09f472554eb62ec48da138a94d96fd7f53c563b7` with `pyproject.toml` version
  `0.10.2`.
- Installed MPS environment:
  `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-mps/bin/python`
  reports `jax==0.10.0`, `jaxlib==0.10.0`, and `jax-mps==0.10.1`. The installed
  package is not version-identical to the local checkout, so source-line claims
  are implementation evidence for the planned patch target, not proof that the
  currently installed wheel has the same exact source.
- `jax-mps` README states the backend is a PJRT plugin that lowers JAX programs
  to StableHLO, maps StableHLO operations to MLX, and executes MLX operations on
  Apple Silicon GPU (`README.md:41-45`, `README.md:100-118`).
- `control_flow.cc` implements `WhileLoopPrimitive` as a custom MLX primitive
  with compiled condition/body functions and per-step evaluation
  (`control_flow.cc:142-176`).
- The same source comment states the primitive must run on the CPU stream for
  orchestration; body and condition dispatch GPU kernels, then
  `mlx::core::eval()` flushes/synchronizes GPU work before reading the scalar
  condition (`control_flow.cc:147-156`).
- The loop implementation evaluates the initial condition with
  `mlx::core::eval(initCond[0])` and reads it via `.item<bool>()`
  (`control_flow.cc:225-230`).
- Each loop trip evaluates the combined body+condition graph with
  `mlx::core::eval(combined)` and then reads `combined[0].item<bool>()`
  (`control_flow.cc:232-247`).
- The StableHLO while handler explicitly selects the CPU stream for this
  orchestration (`control_flow.cc:495-500`).
- `pjrt_executable.cc` reports executable serialization as unimplemented
  (`pjrt_executable.cc:105-106`).
- `PJRT_LoadedExecutable_Execute` currently executes synchronously and returns an
  already-ready event (`pjrt_executable.cc:212-248`).
- The repo's `jax_mps_smoke` backend policy is float32 smoke:
  `requires_x64=False`, `runtime_dtype="float32"`, `host_dtype="float32"`,
  `tolerance_tier="float32_smoke"`, and policy-level
  `default_optimizer_backend="scipy"` (`src/simsopt/backend/runtime.py:327-342`).
- The public single-stage and Stage 2 CLIs resolve `--backend jax` to the outer
  optimizer backend `scipy-jax` when the user does not pass an explicit
  optimizer backend
  (`examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4726-4729`,
  `:7996-8005`;
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:776-779`,
  `:803-808`). The MPS slowdown plan concerns that public/explicit `scipy-jax`
  outer lane, not the Boozer policy's internal `scipy` default.

### Official Documentation Anchors

- JAX `lax.while_loop` documentation states it lowers to a single WhileOp:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html>
- JAX async-dispatch documentation describes asynchronous dispatch and the need
  for `block_until_ready()` when timing or forcing completion:
  <https://docs.jax.dev/en/latest/async_dispatch.html>
- MLX lazy-evaluation documentation states computation is deferred until
  evaluation, evaluation has fixed overhead, and scalar extraction with
  `.item()` triggers evaluation:
  <https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html>
- MLX compile documentation is the baseline for what `mx.compile` can stage,
  cache, and restrict around evaluation and side effects:
  <https://ml-explore.github.io/mlx/build/html/usage/compile.html>
- MLX custom-extension documentation is the fallback path if generic control
  flow cannot be made fast enough:
  <https://ml-explore.github.io/mlx/build/html/dev/extensions.html>
- OpenXLA PJRT API documentation is the contract source for async events and
  executable behavior:
  <https://openxla.org/xla/pjrt/cpp_api_overview>
- StableHLO specification is the contract source for while-loop semantics:
  <https://openxla.org/stablehlo/spec>

### Prior Run Evidence

- The local MPS 3-iteration smoke reached finite initial target-lane value and
  gradient. Its persisted progress stopped at `phase1_attempt_0_started` with
  `event_elapsed_s = 187.53986487499787`; no later optimizer progress was
  written before the process was stopped.
- That run's artifact root was
  `.artifacts/jax_mps_scipyjax_maxiter3_withseed_20260530T005149Z`.
- The same run had finite initial objective evidence:
  `value = 1.1327383518218994`, `grad_inf_norm = 4.2012939453125`, and
  `event_elapsed_s = 187.501`.
- No `results.json` or `REJECTED.json` was produced before the run was killed.
- A current process scan does not show a process using the
  `jax_mps_scipyjax_maxiter3_withseed_20260530T005149Z` output root.

### Design Classification

- Generic `jax-mps` runtime changes are at least Tier 3 under the local software
  design rules because they alter observable PJRT backend behavior for all JAX
  callers.
- A SIMSOPT-only custom primitive or solver rewrite is Tier 2/Tier 3 depending
  on whether the public runtime policy or objective contract changes.
- Required gates: root-cause measurement, design-it-twice, API evolution review,
  correctness tests, rollback/fallback behavior, and benchmark evidence before
  promotion.

## Rationale

The slow MPS behavior is most likely dominated by a stack mismatch: JAX lowers
solver loops to StableHLO while loops, but the local `jax-mps` implementation
orchestrates those loops on the host and synchronizes each trip to read a scalar
condition. Official JAX documentation says the loop is a single WhileOp in the
JAX program, while MLX documentation explains why repeated `eval` and `.item()`
calls have fixed overhead. The local source shows that those calls sit directly
inside the per-trip while loop.

The first engineering step is therefore not a broad SIMSOPT rewrite. It is a
small, measurable `jax-mps` profiling patch that quantifies trip count and time
spent in condition evaluation, body evaluation, scalar reads, and PJRT execute.
Only after that measurement should we optimize.

Design-it-twice:

1. Generic backend fix: detect provably safe while-loop forms and execute more
   work per host synchronization, or move loop semantics closer to device-side
   execution where MLX allows it. This is the preferred path because it improves
   all JAX programs that hit the same backend gap.
2. Workload-specific fix: implement a SIMSOPT-focused MLX/Metal custom extension
   for the Newton/GMRES/Boozer hot path, or refactor the MPS smoke to a
   fixed-iteration masked solver with explicit float32 policy. This is only
   acceptable if generic while-loop performance is blocked by MLX API limits.

## Assumptions

- The local `jax-mps` checkout may be ahead of the installed wheel; either
  reinstall the local checkout into `.conda/jax-mps` or extract the installed
  source before attributing exact source-line behavior to the current wheel.
- The SIMSOPT Boozer/Newton/GMRES objective lowers to one or more
  `stablehlo.while` regions that pass through `HandleWhile`.
- The dominant wait is per-trip host synchronization rather than only cold Metal
  shader compilation.
- CPU and CUDA reference lanes remain available for correctness and performance
  comparison.
- MLX does not currently expose a generic arbitrary StableHLO device-side while
  primitive equivalent to XLA-CUDA; this must be rechecked before implementation.

## Implementation Plan

1. Baseline the current behavior.
   - [ ] Resolve the installed/local version mismatch: installed
     `jax-mps==0.10.1`, local checkout `0.10.2`. Either reinstall the local
     checkout into `.conda/jax-mps` for patch testing or extract the installed
     wheel source before making exact current-runtime claims.
   - [ ] Add a minimal `jax-mps` control-flow microbenchmark covering
     `lax.while_loop`, nested `lax.while_loop`, `lax.scan`, fixed trip count,
     dynamic convergence, and `jax.value_and_grad`.
   - [ ] Run the microbenchmark on CPU and MPS with loop trip counts such as
     1, 10, 100, and 1000.
   - [ ] Record cold-start and warm-start timings separately so compile cost
     does not hide steady-state loop behavior.
   - [ ] Re-run the SIMSOPT 3-iteration MPS smoke with a hard timeout and keep
     `outer_optimizer_progress.json`, process logs, and process samples.

2. Add `jax-mps` runtime instrumentation.
   - [ ] Add an opt-in environment variable such as `JAX_MPS_PROFILE_WHILE=1`.
   - [ ] In `WhileLoopPrimitive::eval_impl`, count initial condition evals,
     body+condition evals, scalar condition reads, final evals, and loop trips.
   - [ ] Time `compiledCond_`, `compiledBodyCond_`, `mlx::core::eval`, and
     `.item<bool>()` separately.
   - [ ] Emit structured JSON lines to stderr or a path set by an environment
     variable; do not change default runtime output.
   - [ ] Add PJRT execute timing around `mlx_executable->Execute(inputs)` and
     ready-event creation.
   - [ ] Add focused tests proving instrumentation is off by default and does
     not alter numerical outputs when enabled.

3. Prototype the generic control-flow fast path.
   - [ ] Inspect the StableHLO while regions from the microbenchmark and the
     SIMSOPT objective to identify induction-counter and max-iteration patterns.
   - [ ] Build a detector for statically bounded loops where a maximum trip
     count and loop-carried values can be proven from StableHLO.
   - [ ] Prototype chunked execution for safe loops: run up to `K` body trips
     before a host condition sync, while preserving exact early-exit behavior at
     chunk boundaries.
   - [ ] Use strict fallback to the existing per-trip implementation when the
     detector cannot prove safety.
   - [ ] Test exact result parity for early exit, zero-trip loops, max-trip
     loops, nested loops, gradients, and side-effect-free loop bodies.
   - [ ] Benchmark `K` values to find whether chunking reduces MPS wall time
     without excessive graph growth or memory pressure.

4. Evaluate PJRT secondary improvements.
   - [ ] Design an async-ready-event path consistent with the OpenXLA PJRT C++
     API contract.
   - [ ] Add tests for `block_until_ready()`, host transfer, and error
     propagation before enabling async execution.
   - [ ] Implement executable serialization/fingerprint only after the
     correctness suite is stable.
   - [ ] Measure cold compile, warm compile, and repeated process startup
     separately to quantify serialization value.
   - [ ] Keep async dispatch and serialization as secondary work unless the
     while-loop instrumentation shows per-trip sync is not dominant.

5. Decide on a SIMSOPT-specific fallback only if needed.
   - [ ] If the generic fast path does not move the 3-iteration smoke, extract
     the dominant Boozer/Newton/GMRES StableHLO or JAXPR region.
   - [ ] Decide whether the hot path can be represented as an MLX custom
     extension while preserving the required VJP/JVP behavior.
   - [ ] Compare two local alternatives: a custom MLX/Metal extension versus an
     MPS-only fixed-iteration masked solver for smoke testing.
   - [ ] Keep any SIMSOPT policy change explicit, typed, and fail-loud for
     unsupported production modes.
   - [ ] Document why the generic backend path failed before accepting a
     SIMSOPT-only customization burden.

6. Integrate, validate, and split patches.
   - [ ] Install the patched `jax-mps` into the repo's MPS environment.
   - [ ] Run the control-flow microbenchmark before and after the patch.
   - [ ] Run the SIMSOPT 3-iteration MPS smoke and compare progress against the
     previous 27-minute stall.
   - [ ] Run the CPU reference smoke and compare objective/gradient within the
     existing float32-smoke tolerance.
   - [ ] Split patches into instrumentation, generic runtime improvement,
     secondary PJRT improvements, and SIMSOPT policy/documentation changes.
   - [ ] Keep rollback simple: disabling the new fast path must restore the
     current `WhileLoopPrimitive` behavior.

## Validation Plan

- [ ] `git -C /Users/suhjungdae/code/opensource/jax-mps status --short`
- [ ] `git status --short` in `/Users/suhjungdae/code/columbia/simsopt-jax`
- [ ] `.conda/jax-mps/bin/python -m pip show jax-mps`
- [ ] `.conda/jax-mps/bin/python -c 'import importlib.util; print(importlib.util.find_spec("jax_plugins.mps"))'`
- [ ] Focused `jax-mps` tests for while-loop result parity and instrumentation
  default-off behavior.
- [ ] In `/Users/suhjungdae/code/opensource/jax-mps`, run
  `uv run python scripts/run_jax_tests.py` or a documented focused subset if the
  full upstream JAX suite is too expensive.
- [ ] Control-flow microbenchmark: CPU and MPS timings for cold and warm runs.
- [ ] SIMSOPT 3-iteration MPS smoke with a hard timeout and artifact capture.
- [ ] Check that `outer_optimizer_progress.json` progresses beyond the previous
  `phase1_attempt_0_started` stall, or records a clear failure artifact.
- [ ] Check that initial objective and gradient remain finite.
- [ ] Compare CPU reference objective/gradient to MPS within float32-smoke
  tolerance.
- [ ] Verify no matching runaway benchmark processes remain after each run:
  `ps -axo pid,etime,command | rg 'jax_mps_scipyjax|JAX_PLATFORMS=mps|SIMSOPT_BACKEND_MODE=jax_mps_smoke'`.
- [ ] `git diff --check` for every edited checkout.

## Risks and Mitigations

- Risk: A generic while-loop fast path changes early-exit semantics.
  Mitigation: Only enable it when the detector proves the loop shape, and keep a
  strict fallback to the current implementation.

- Risk: Chunked execution reduces sync overhead but grows MLX graphs enough to
  increase memory use or compile time.
  Mitigation: Benchmark multiple chunk sizes and require cold/warm timing plus
  memory evidence before selecting a default.

- Risk: MLX public APIs cannot express arbitrary device-side StableHLO while
  semantics.
  Mitigation: Treat that as a measured blocker and decide explicitly between
  custom MLX extension work and SIMSOPT-only smoke refactoring.

- Risk: Async PJRT execution breaks readiness, host transfer, or error
  propagation expectations.
  Mitigation: Keep async work behind focused `block_until_ready` and transfer
  tests, and do not combine it with while-loop changes in the same patch.

- Risk: Float32 MPS results look fast but are not production-valid for Boozer
  convergence.
  Mitigation: Keep production x64 gates on CPU/CUDA and label MPS as smoke-only
  until numerical parity is proven at the required resolution. Keep the float64
  exclusion tied to the repo's `jax_mps_smoke`/`jax-mps` lane, not to a broad
  statement about every MLX dtype.

- Risk: Backend changes create a long-lived fork that is hard to maintain.
  Mitigation: Prefer upstreamable instrumentation and generic fast paths; require
  a written decision before accepting SIMSOPT-specific runtime code.

## Completion Criteria

- [ ] The current MPS slowdown is explained by measured data rather than
  inference alone.
- [ ] `jax-mps` emits opt-in while-loop profile data with no default behavior
  change.
- [ ] A generic fast path either improves the control-flow microbenchmark and
  SIMSOPT 3-iteration smoke, or is rejected with concrete MLX/API evidence.
- [ ] Focused `jax-mps` control-flow tests pass.
- [ ] SIMSOPT CPU and MPS smoke checks produce finite objective/gradient
  artifacts within the defined float32-smoke policy.
- [ ] The previous 3-iteration stall is either materially improved or replaced
  by a clear, fail-loud rejection artifact.
- [ ] A decision is recorded: upstream generic `jax-mps` patch, local fork,
  SIMSOPT-specific custom primitive, or no MPS investment.

## Open Questions

- Does the installed `jax-mps` package exactly match
  `/Users/suhjungdae/code/opensource/jax-mps`? Current evidence says no by
  package version (`0.10.1` installed vs `0.10.2` checkout), but exact source
  diff still needs extraction or reinstall-based testing.
- Which specific SIMSOPT JAXPR or StableHLO regions account for the largest
  while-loop trip counts?
- What speedup threshold justifies maintaining a local `jax-mps` fork?
- Can MLX custom extensions support the required differentiation path for the
  Boozer/Newton/GMRES hot path?
- Is MPS worth optimizing beyond smoke testing while production Boozer
  convergence still requires float64?
- Should this work be proposed upstream first, or carried locally until the
  SIMSOPT benchmark demonstrates value?
