# TORAX-Style Host-Controlled L-BFGS Kernelization Plan

## Purpose

Define the production fix for the single-stage JAX optimizer path after the
`lbfgs-ondevice` A100 production canary showed that a stepwise L-BFGS driver can
still behave like an expensive macro-kernel when it encloses the full
single-stage value/gradient pipeline.

This plan converts the TORAX compile-boundary lesson into executable SIMSOPT
work: keep optimizer, retry, and line-search control on the host; compile only
bounded, static-shape value/residual/Jacobian/gradient kernels; and prove compile
and memory stability before rerunning the six-seed production matrix.

## Goals

- Add a production single-stage JAX path whose compiled units are small,
  static-signature cached kernels, not a full optimizer macro-step graph.
- Preserve precision parity against the existing CPU/C++/Python and host-driven
  SciPy/JAX references on curated warm-start seeds.
- Prove same-shape dynamic values do not trigger recompilation, and optimizer
  step count does not grow compile count or steady memory after warmup.
- Produce A100 80 GB evidence for walltime, precision, host RSS, and GPU memory
  on the same matrix previously used for `scipy-jax`.

## Non-Goals

- Do not try to make the full single-stage outer L-BFGS loop a single on-device
  graph.
- Do not make `lbfgs-ondevice` the production single-stage route until it passes
  the same compile-count, memory, precision, and walltime gates.
- Do not replace the existing CPU fallback or the current scalar/private
  `lbfgs-ondevice` behavior tests.
- Do not count reporting-only hardware/status metrics as optimizer walltime
  unless every compared lane pays the same work.

## Current Context

- Current checkout for this plan: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean`
  at `e87fac35a`, with an existing dirty worktree. This plan is a new isolated
  docs file.
- TORAX uses immutable hash-by-value static arguments for JIT/persistent-cache
  safety in `/Users/suhjungdae/code/opensource/torax/torax/_src/static_dataclass.py:16`.
- TORAX splits static `JAX_STATIC` fields from dynamic pytree values in
  `/Users/suhjungdae/code/opensource/torax/torax/_src/torax_pydantic/model_base.py:78`.
- TORAX bounds differentiable loops with static `max_steps` and `jax.lax.scan`
  in `/Users/suhjungdae/code/opensource/torax/torax/_src/jax_utils.py:237`.
- TORAX scopes solver compilation to residual/loss/one-step solve kernels with
  explicit `static_argnames` in
  `/Users/suhjungdae/code/opensource/torax/torax/_src/fvm/residual_and_loss.py:214`
  and `/Users/suhjungdae/code/opensource/torax/torax/_src/fvm/optimizer_solve_block.py:42`.
- TORAX avoids differentiating through full iterative solves where that is the
  wrong graph: Newton uses implicit differentiation/custom root because
  reverse-mode through the `while_loop` would be inefficient in
  `/Users/suhjungdae/code/opensource/torax/torax/_src/solver/jax_root_finding.py:112`.
- TORAX tests compile behavior directly with `get_number_of_compiles` in
  `/Users/suhjungdae/code/opensource/torax/torax/_src/jax_utils.py:147`, including
  same-shape dynamic updates staying at one compile in
  `/Users/suhjungdae/code/opensource/torax/torax/_src/config/tests/build_runtime_params_test.py:260`.
- TORAX disables callback-based errors for persistent-cache testing because host
  callbacks cannot serialize into the persistent compilation cache in
  `/Users/suhjungdae/code/opensource/torax/torax/tests/persistent_cache_test.py:59`.
- JAX's documented `jax.jit` behavior matches this plan's cache boundary:
  static arguments are part of the compilation contract and can force recompiles
  when changed, while the persistent compilation cache is a disk cache for
  compatible compiled programs. JAX GPU dispatch is asynchronous, so benchmark
  timings must use completed subprocess boundaries or explicit synchronization
  rather than raw dispatch timestamps.
- The current SIMSOPT `lbfgs-ondevice` stepwise kernel is still named
  `lbfgs_private_macro_step_solver` and is compiled as `jax.jit(run)` in
  `src/simsopt_jax/geo/optimizers/private/_lbfgs.py:262` and
  `src/simsopt_jax/geo/optimizers/private/_lbfgs.py:297`.
- That macro-step calls a value/gradient callable; for single-stage traceable
  objectives, the value/gradient path includes `_value_and_grad_for`, which runs
  the forward solve and then the total-gradient kernel in
  `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:1388`.
- A solved-state value/gradient kernel already exists as a useful seed:
  `_solved_state_value_and_grad_for` in
  `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:1454` evaluates
  objective and adjoint gradient from a solved Boozer state and factors.
- The current `host-jax` path already uses SciPy/L-BFGS host control for the
  outer optimizer and builds a solved-state value/gradient factory in
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:15144`.
  That is groundwork, not a complete production kernel bundle: failure guards,
  static signature ownership, residual/Jacobian/factor kernel splitting, and a
  current six-seed A100 production matrix are still missing.
- The latest synced A100 `lbfgs-ondevice` production failure evidence is
  `.artifacts/runpod_lbfgsondevice_prod1500_50_a100_failure_20260619T2235Z`,
  with the writeup in
  `.artifacts/lbfgsondevice_prod1500_50_failure_deepdive_20260619.md`. Seeds
  1-3 all exited 124 with no result JSON; seeds 2 and 3 reported
  `line_search_failed`, `ls_status=-9`, zero accepted iterations, and peak
  sampled GPU memory around 20 GiB, so the immediate failure is not A100 VRAM.

## Rationale

The failed direction was adding more knobs around a fused optimizer path. TORAX
shows the better abstraction boundary: static configuration chooses a cached
kernel bundle; dynamic values flow through explicit JAX array arguments; host
code owns iteration, failure policy, logging, and retry.

Two implementation options were considered:

- Extend the existing `lbfgs-ondevice` macro-step. This preserves the current
  public method name but keeps line search and failed-step retry coupled to a
  large compiled value/gradient invocation. The A100 failure mode is exactly that
  coupling, so this is not the production fix.
- Build or extend a host-controlled `host-jax` / `kernelized-jax` backend.
  SciPy/Python owns L-BFGS and line search. GPU kernels do the expensive
  residual, solve, value, and adjoint-gradient math. This matches TORAX's
  compilation discipline and gives observable compile/memory gates.

Choose the second option for production single-stage. Treat existing `host-jax`
as the starting point if the backend name is retained, but promote only after it
owns the missing static kernel bundle and production gates. Keep
`lbfgs-ondevice` as a separate scalar/private optimizer lane until it earns
production evidence.

## Assumptions

- The expensive physics math can be expressed through static-shape kernels whose
  signatures are keyed by resolution, mode counts, Boozer mode, masks,
  `optimize_G`, coil grouping, dtype, backend, and objective options.
- Dynamic coil arrays, coil DOFs, solved Boozer state, linear-solve factors, and
  runtime values can be passed as explicit JAX array arguments rather than
  captured as cache-busting closure state.
- Existing host-driven `scipy-jax` parity machinery can be reused for benchmark
  comparison, but may need a new target lane label for the kernelized path.
- The next production validation target is A100 80 GB, not H100.

## Implementation Plan

1. Freeze the failure boundary and fail fast.
   - [x] Record the latest A100 `lbfgs-ondevice` failure artifact path, seed
         statuses, `ls_status`, timeout, and peak GPU memory in this plan and
         `.artifacts/lbfgsondevice_prod1500_50_failure_deepdive_20260619.md`.
         Peak RSS was not present in the synced failure artifact, so do not
         claim an RSS number for that failed matrix.
   - [ ] Add a single-seed canary gate that aborts a production matrix before
         launching all seeds if initial value/grad or first optimizer step
         exceeds a fixed walltime budget or repeats the same line-search failure
         anchor.
   - [ ] Add a production-route warning or guard: full single-stage
         `lbfgs-ondevice` is experimental until the gates in this plan pass.

2. Define the static kernel signature and cache owner.
   - [ ] Add a small immutable kernel signature type for single-stage/Boozer
         kernels. Include only static compile-affecting fields: resolution,
         surface mode counts, Boozer LS/exact mode, active masks, `optimize_G`,
         coil grouping, objective option signature, dtype, and backend.
   - [ ] Hoist any missing single-stage kernel-bundle construction into
         `BoozerSurfaceJAX` or a dedicated factory near
         `surface_objectives_traceable.py`, not inside per-objective calls. The
         existing `_traceable_runtime_entry_cache` and
         `_traceable_solved_state_value_and_grad_entry_cache` are partial cache
         owners, but they are not yet the production bundle boundary described
         here.
   - [ ] Make all dynamic state explicit kernel arguments: coil DOFs, coil arrays,
         solved state, linear-solve factors, current residual state, and any
         runtime scalar arrays.
   - [ ] Extend the existing same-shape reuse coverage in
         `tests/geo/test_boozersurface_jax.py::TestUpstreamBoozerSurfaceJAX::test_host_jax_kernel_bundle_reuses_compiled_kernels_same_shape`
         with an intentional static-signature-change case that creates exactly
         one distinct bundle.

3. Split the computational kernels.
   - [x] Identify current residual/Jacobian/factor kernels in the
         `BoozerSurfaceJAX` penalty bundle and solved-state value/gradient seed.
   - [ ] Promote those pieces into the static-signature production bundle with
         explicit dynamic-state arguments: `residual(x, dynamic_state)`,
         `jacobian(x, dynamic_state)` or equivalent factorization, and
         `factor_apply(...)`.
   - [x] Keep a solved-state value/gradient kernel based on
         `_solved_state_value_and_grad_for`, so the outer optimizer does not
         differentiate through all Boozer iterations.
   - [ ] Add direct tests that the production solved-state value/gradient path
         never calls the full `_value_and_grad_for` forward-solve graph.
   - [ ] Add a `solve_boozer_state(...)` host helper that calls residual,
         Jacobian, and solve kernels step by step and returns solved state,
         factors, convergence metadata, and failure reason.
   - [ ] Keep hardware status, final reporting, and optional diagnostics outside
         the timed optimizer-core path unless every benchmark lane pays for them.

4. Add the host-controlled optimizer backend.
   - [x] Confirm `host-jax` already resolves to a host-controlled SciPy/L-BFGS
         contract for the single-stage outer loop.
   - [x] Confirm the current host-JAX adapter can solve Boozer state on the host,
         call a solved-state value/gradient kernel, and return `(value, grad)` to
         SciPy/L-BFGS.
   - [ ] Promote that path to the production kernelized lane only after the
         static bundle, fail-fast, compile-count, memory, and six-seed A100 gates
         pass.
   - [x] Keep line search, convergence, and callback-stop on the Python host for
         `host-jax`.
   - [ ] Keep retry/failure policy on the Python host with a fail-fast rule for
         repeated zero-iteration line-search failures from the same anchor.
   - [ ] On line-search failure, return one explicit failure event and stop or
         switch policy; do not relaunch the same expensive compiled step from the
         same failed anchor.
   - [ ] Preserve CPU/C++/Python and `scipy-jax` route behavior until the new lane
         passes parity and memory gates.

5. Add compile-count and memory gates.
   - [x] Reuse current same-shape `_cache_size()` coverage for Boozer penalty
         kernels and compile-log coverage for `lbfgs-ondevice` diagnostics.
   - [ ] Add production-lane `_cache_size()` / compile-log tests proving same
         shape with different dynamic values does not compile again.
   - [ ] Add a new-shape test proving exactly one additional compile for one
         intentional static signature change.
   - [ ] Add an optimizer-step test proving compile event count and cache miss
         count do not grow with additional optimizer iterations after warmup.
   - [ ] Add host RSS and GPU memory sampling gates proving no steady growth per
         optimizer step after warmup.
   - [ ] Add a persistent-cache compatibility check that no cached kernel uses
         host callbacks or debug callbacks inside the serializable kernel body.

6. Run the production comparison matrix.
   - [ ] Use the curated existing converged artifacts from
         `/Users/suhjungdae/code/columbia/autoresearch/runs` for the same six
         warm-start seeds used by the `scipy-jax` comparison.
   - [ ] Run CPU/C++/Python reference, `scipy-jax`/host reference, JAX CPU, and
         JAX GPU kernelized lanes against identical seed inputs and Stage 2 field
         artifacts.
   - [ ] For every lane, collect precision parity, optimizer-core walltime,
         process walltime, host RSS, GPU peak memory where applicable, compile
         count, cache misses, and failure status.
   - [ ] Run on A100 80 GB only for the production GPU gate unless a separate
         H100 exploratory run is explicitly requested.

## Validation Plan

- [ ] `python -m py_compile src/simsopt_jax_adapters/geo/surface_objectives_traceable.py src/simsopt_jax_adapters/geo/boozer_surface.py src/simsopt_jax/geo/optimizers/optimizer.py`
- [ ] `python -m pytest tests/geo/test_boozersurface_jax.py -k "host_jax_kernel_bundle_reuses_compiled_kernels_same_shape or solved_state"`
- [ ] `python -m pytest tests/integration/test_single_stage_physics_parity.py -k "host_jax_compile_gate or host_jax_memory_gate or host_jax_adapter_uses_solved_state_kernel_without_legacy_objective or host_jax_adapter_builds_solved_state_kernel_after_boozer_solve"`
- [ ] `python -m pytest tests/test_gpu_transfer_guard_harness.py -q`
- [ ] `python -m pytest tests/test_lbfgs_ondevice_compile_shape.py tests/geo/test_boozersurface_jax_private.py -k "lbfgs_ondevice or macro_step"`
- [ ] Add a benchmark JSON gate that reports `steady_compile_event_count_growth == 0`,
      `steady_cache_miss_count_growth == 0`, `peak_steady_gpu_memory_growth_mb == 0`,
      and `peak_steady_rss_growth_mb == 0` for the kernelized lane.
- [ ] Run the A100 six-seed matrix and write a summary artifact comparing
      precision, optimizer-core walltime, full walltime, RSS, GPU memory, compile
      counts, and failure reasons across CPU/C++/Python, `scipy-jax`, JAX CPU,
      and JAX GPU.

## Risks and Mitigations

- Risk: Dynamic arrays remain captured in closures, causing recompilation by
  object identity or hidden shape drift.
  Mitigation: make dynamic values explicit args and add same-shape/different-value
  compile-count tests.

- Risk: Host-controlled line search increases host/device transfer overhead.
  Mitigation: transfer only scalar value and gradient vector per trial; keep
  residual/Jacobian/factor/gradient math on device; report optimizer-core and
  reporting walltime separately.

- Risk: Reporting and hardware-status work hides optimizer speedups.
  Mitigation: split timing into initial status, core optimizer, final reporting,
  compile, and full process walltime.

- Risk: Persistent-cache writes silently fail because debug/error callbacks are
  inside kernels.
  Mitigation: mirror TORAX's callback exclusion policy for cached kernels and add
  a persistent-cache smoke test.

- Risk: The new backend becomes a parallel implementation with drift from
  `scipy-jax`.
  Mitigation: keep CPU reference parity and six-seed comparison mandatory before
  promotion; leave existing routes untouched until promotion.

## Completion Criteria

- [ ] A kernel bundle is cached by a stable static signature and receives dynamic
      coil/state values as explicit arguments.
- [ ] The production single-stage JAX lane does not call a compiled optimizer
      macro-step that encloses full Boozer solve, line search, retry, and
      reporting.
- [ ] Compile-count gates prove one compile per static shape and no compile growth
      with optimizer steps.
- [ ] Memory gates prove no steady host RSS or GPU memory growth after warmup.
- [ ] Six-seed A100 matrix reports precision parity, walltime, and memory against
      CPU/C++/Python and `scipy-jax` references.
- [ ] Documentation states exactly which backend is production-supported and which
      on-device L-BFGS paths remain experimental.

## Open Questions

- Should the production backend name stay `host-jax`, or should this become a
  distinct `kernelized-jax` lane until promotion?
- Which static signature fields are required for the first production seed set,
  and which can be postponed without risking hidden recompiles?
- Should `lbfgs-ondevice` be hard-blocked for production single-stage runs, or
  left available with a fail-fast canary and explicit warning?
- What exact walltime budget should the single-seed A100 canary use before it
  aborts the full six-seed matrix?
