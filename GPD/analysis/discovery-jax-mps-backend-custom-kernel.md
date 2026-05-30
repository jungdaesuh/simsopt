# Discovery: JAX-MPS Backend/Custom-Kernel Feasibility

Date: 2026-05-30
Depth: medium
Topic: Whether the SIMSOPT `scipy-jax` MPS slowdown requires backend/custom-kernel work
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Related backend checkout: `/Users/suhjungdae/code/opensource/jax-mps`

## Summary

Recommendation: Treat this as a real backend/custom-kernel project.

The evidence supports the narrower claim that the workload is not inherently impossible on a GPU. CUDA results show that a mature GPU stack can win. The MPS problem is that the current jax-mps/MLX path exposes SIMSOPT's dynamic Newton/GMRES control flow as repeated host/device synchronization.

Small fixes can help diagnostics and synthetic loops, but the matched SIMSOPT smoke did not complete after fixed-trip backend chunking. The next credible implementation work is either a jax-mps dynamic control-flow backend improvement or a SIMSOPT/MLX custom primitive that owns the hot solver loop internally.

Confidence: high.

## Scope

Included:

- Official JAX and MLX documentation relevant to control flow, async dispatch, lazy evaluation, scalar reads, and MLX extensions.
- Local jax-mps backend source inspection.
- Local SIMSOPT source inspection for Boozer/Newton/GMRES control flow.
- Prior local prototype and run artifact outcomes.

Excluded:

- New long-running SIMSOPT benchmarks.
- New CUDA measurements.
- Implementing a production backend fix.

## Sources

Official docs:

- JAX `lax.while_loop`: https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
- JAX async dispatch: https://docs.jax.dev/en/latest/async_dispatch.html
- MLX lazy evaluation: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX C++ extensions: https://ml-explore.github.io/mlx/build/html/dev/extensions.html

Local source anchors:

- `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/ops/control_flow.cc`
- `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/pjrt_executable.cc`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py`
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py`

Local run artifact:

- `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax_mps_scipyjax_maxiter3_patched_withseed_20260530T025720Z/RUNTIME_FIX_DECISION.md`

## Findings

### 1. JAX intends compiled loop execution and async device dispatch

JAX documents `lax.while_loop` as staged control flow. JAX also documents asynchronous dispatch, including the need for `block_until_ready()` when benchmarking. That matters because a backend that turns loop predicates into host reads loses the latency-hiding behavior expected from a mature JAX device runtime.

### 2. MLX lazy evaluation makes scalar control flow expensive when forced frequently

MLX documents lazy evaluation and explicit evaluation triggers. Scalar extraction through `.item()` forces evaluation. The MLX docs also call out scalar-array control flow as inefficient when frequent.

That matches the observed failure mode: SIMSOPT's solver loops repeatedly decide whether to continue based on dynamic scalar convergence state.

### 3. jax-mps currently executes dynamic while control flow through host orchestration

The local jax-mps `control_flow.cc` implementation orchestrates the dynamic/default while path from the CPU stream. It evaluates MLX arrays and reads the condition through `.item<bool>()`. The implementation comment states that the GPU stream path would deadlock because evaluation synchronizes the GPU queue.

The local `pjrt_executable.cc` path also marks execution as synchronous for now, creating a ready event after execution.

So the backend exposes every dynamic while trip as a drain/read/relaunch boundary.

### 4. SIMSOPT's hot path is dynamic Newton/GMRES control flow

The SIMSOPT source routes Boozer exact solve work through operator GMRES and traceable Newton paths. The relevant paths use `jax.lax.while_loop` with dynamic convergence predicates based on iteration counts, residual norms, status flags, and solver info.

The strict scalar objective and gradient are cached and JIT compiled, but the compiled region still contains the dynamic solver loops.

### 5. The fixed-trip backend prototype improved microbenchmarks but not the real SIMSOPT run

A local jax-mps prototype added while-loop profiling and fixed-trip chunking. It passed focused tests and improved synthetic loops:

- `while_fixed_trip`, 100 trips: `1.7x`
- `scan_fixed_trip`, 100 trips: `4.0x`
- `nested_while`, 100 trips: `3.5x`

The matched SIMSOPT `maxiter=3` smoke still timed out at 600 seconds. It reached a finite initial objective and gradient, but did not produce `results.json` or `REJECTED.json`.

Checkpoint timings regressed:

- `initial_hardware_status_returned`: `131.31 s` patched vs `113.53 s` previous.
- `target_lane_initial_objective_value_and_grad_returned`: `209.42 s` patched vs `187.49 s` previous.

This is the disconfirming check against "small backend tweak is enough." It helped where the loop was fixed-trip, but not where SIMSOPT is dynamic-convergence dominated.

## Method Comparison

### Python-level tuning

Expected impact: low.

Python-level tuning can reduce setup overhead or make measurement cleaner. It cannot remove backend scalar reads inside a compiled dynamic solver loop.

### JIT/cache improvements

Expected impact: medium for cold-start overhead, low for steady-state dynamic-loop cost.

Compile cache work can matter for short runs, but it does not address per-trip runtime synchronization once the objective-and-gradient executable is running.

### Fixed-trip chunking

Expected impact: high for fixed-trip loops, low for SIMSOPT's current dynamic loops.

The local prototype supports this: microbenchmarks improved; the SIMSOPT smoke did not.

### Backend dynamic control-flow implementation

Expected impact: high.

This targets the identified mechanism directly: keep the convergence predicate and loop continuation on device, or at least reduce host boundaries while preserving semantics.

### Custom SIMSOPT/MLX primitive

Expected impact: high, with higher implementation risk.

A custom primitive could own the Boozer/Newton/GMRES block internally and return only final objective/gradient data to the JAX boundary.

### Algorithm restructuring

Expected impact: uncertain.

Fixed-iteration masked or batched solvers could give the current backend more parallel width, but numerical parity and convergence semantics become the hard part.

## Recommendation

Proceed only if the project is scoped as one of:

1. jax-mps backend work for dynamic `stablehlo.while` execution.
2. MLX/C++/Metal custom primitives for the SIMSOPT hot solver block.
3. A scientific rewrite of the SIMSOPT solver into fixed-iteration or batched forms with strict CPU/CUDA/MPS parity gates.

Do not represent environment flags, host optimizer changes, or fixed-trip-only chunking as a full solution.

## Validation Gates

A future implementation should not be called successful until:

1. A GMRES-like dynamic convergence microbenchmark improves on MPS without changing CPU parity.
2. The SIMSOPT `scipy-jax` MPS `maxiter=3` smoke completes.
3. Matched checkpoint timings improve against the previous artifact.
4. Objective and gradient parity are checked against CPU JAX.
5. The same path is compared against CUDA at matched problem size if CUDA access is available.

## Open Questions

- Whether MLX exposes enough low-level control to implement a true device-resident dynamic while primitive without upstream MLX changes.
- Whether SIMSOPT's exact Newton/GMRES hot block is better handled as one fused custom primitive or several smaller primitives.
- Whether a fixed-iteration masked solver is scientifically acceptable for this workload.

## Final Classification

This is a real backend/custom-kernel project.

The classification is based on the source-level mechanism, official docs, and the failed SIMSOPT validation of the smaller fixed-trip backend patch.
