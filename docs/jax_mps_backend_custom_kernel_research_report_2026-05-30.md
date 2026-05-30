# JAX-MPS Backend/Custom-Kernel Research Report

Date: 2026-05-30
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Related backend checkout: `/Users/suhjungdae/code/opensource/jax-mps`

## Verdict

Yes. For the SIMSOPT `scipy-jax` workload on Apple MPS, closing the observed gap is a real backend/custom-kernel project.

The current evidence does not support treating this as a small Python tuning, environment-variable, or optimizer-routing fix. The bottleneck sits at the execution boundary between JAX control flow, the jax-mps PJRT backend, MLX lazy evaluation, and SIMSOPT's dynamic Newton/GMRES solver loops.

The shortest accurate statement is:

> The workload can run fast on a mature GPU stack, but jax-mps currently exposes solver-loop control flow to the host too often. Fixing that requires backend work or custom kernels that keep the hot convergence loop on device.

## GPD Scope Note

GPD skills were used to route and structure this research task. The local repo is not currently initialized as a GPD project:

- `state.json` is absent.
- `STATE.md` is absent.

The GPD discovery workflow was therefore used in standalone mode. The standalone discovery artifact is:

- `GPD/analysis/discovery-jax-mps-backend-custom-kernel.md`

This document is the repo-facing report under `docs/`, not a completed phase-scoped GPD artifact.

## Claim Evaluated

The claim is not "Apple GPUs cannot run this workload."

The claim is:

1. The CUDA result proves the algorithm can benefit from a GPU when the runtime keeps enough work on device.
2. The MPS result is slow because the current jax-mps/MLX stack cannot keep SIMSOPT's dynamic solver control flow on device.
3. A real fix therefore requires backend/compiler/runtime work, or a custom kernel/primitive that hides the Newton/GMRES loop from Python/PJRT-level scalar polling.

This claim is supported.

## Official Documentation Evidence

### JAX Control Flow And Async Dispatch

JAX documents `jax.lax.while_loop` as staged control flow intended for compiled execution:

- https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html

JAX also documents asynchronous dispatch and the need to use `block_until_ready()` when benchmarking, because normal JAX execution can enqueue device work and let the host continue:

- https://docs.jax.dev/en/latest/async_dispatch.html

This matters because the mature JAX GPU path is designed around queued device execution. If a backend turns every loop predicate into an eager host read, it loses one of JAX's main latency-hiding mechanisms.

### MLX Lazy Evaluation And Scalar Reads

MLX documents lazy evaluation. Operations build a graph until evaluation is forced. Evaluation is forced by calls such as `mx.eval()`, conversion, printing, and scalar extraction through `.item()`.

- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html

The same MLX docs warn that scalar-array control flow can be inefficient when frequent, because those scalar decisions force evaluation.

MLX also documents C++ extension support:

- https://ml-explore.github.io/mlx/build/html/dev/extensions.html

That is important because the natural repair path is not just Python code. MLX has a native extension route for custom operations, and this workload likely needs an operation boundary that owns more of the solver loop internally.

## Local jax-mps Backend Evidence

The jax-mps control-flow implementation confirms the host-synchronization model.

In `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/ops/control_flow.cc`, the while-loop implementation comments describe the current primitive:

- The CPU stream orchestrates the loop.
- Each trip calls compiled body/condition code.
- Each trip runs `mlx::core::eval()`.
- Each trip reads the predicate with `.item<bool>()`.
- The GPU stream path would deadlock because evaluation synchronizes the GPU queue.

The dynamic/default loop path does exactly that:

- Evaluate the initial condition with `mlx::core::eval(initCond[0])`.
- Read the initial predicate with `initCond[0].item<bool>()`.
- For each loop trip, run the combined body/condition.
- Force evaluation with `mlx::core::eval(combined)`.
- Read the next predicate with `combined[0].item<bool>()`.

In `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/pjrt_executable.cc`, PJRT execution is also marked synchronous for now. A ready event is created immediately after execution instead of representing asynchronously queued work.

Those two details are enough to explain why a dynamic convergence loop becomes expensive on MPS:

1. The backend cannot keep the loop predicate entirely on device.
2. The host cannot run ahead and hide dispatch latency.
3. Every solver iteration exposes a drain/read/relaunch boundary.

## SIMSOPT Hot-Path Evidence

The SIMSOPT hot path is not a single large dense GPU kernel. It is a solver stack with dynamic convergence.

In `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/boozersurface_jax.py`:

- Boozer exact linear solves are routed through operator GMRES.
- The manual penalty least-squares path uses `jax.lax.while_loop` with a dynamic condition based on iteration count and residual norm.
- Traceable Newton paths call `newton_exact_traceable(...)` or `_run_newton_polish_for_method(...)` with `newton_maxiter` and `newton_tol`.

In `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/optimizer_jax.py`:

- Levenberg-Marquardt and Newton paths use `lax.while_loop` with dynamic convergence status.
- GMRES is invoked through `jax.scipy.sparse.linalg.gmres`.
- Exact Newton solve paths call `_run_operator_gmres(...)`.

In `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/geo/surfaceobjectives_jax.py`:

- The strict scalar objective and gradient are cached and JIT compiled.
- `BoozerResidualJAX` uses cached strict scalar value-and-gradient evaluation.

So the expensive region is not simply "SciPy is slow" or "L-BFGS-B is on CPU." The outer optimizer is host-side in both CPU and MPS lanes. The differing region is the JAX value-and-gradient evaluation, and inside that evaluation SIMSOPT enters dynamic solver loops.

## Prototype Evidence

A local jax-mps prototype added profiling and a fixed-trip chunking fast path.

Validation for the prototype passed:

- `uv pip install -e .`
- `ruff format`
- `ruff check`
- `python -m py_compile`
- `uv run pytest -q tests/test_control_flow_profile.py`: 4 passed
- `uv run pytest -q tests/test_ops.py -k control_flow`: 96 passed, 32 skipped

Microbenchmarks improved for fixed-trip or effectively fixed-trip loops:

- `while_fixed_trip`, 100 trips: `0.022815458 s` to `0.013659625 s` (`1.7x`)
- `scan_fixed_trip`, 100 trips: `0.028547167 s` to `0.007111916 s` (`4.0x`)
- `nested_while`, 100 trips: `0.488316542 s` to `0.139788625 s` (`3.5x`)

But the SIMSOPT `maxiter=3` smoke did not improve enough to complete:

- Artifact root: `.artifacts/jax_mps_scipyjax_maxiter3_patched_withseed_20260530T025720Z`
- Process result: exit code `124` from a 600 second timeout.
- Last progress event: `objective_evaluation`.
- Initial objective was finite: `1.1327383518218994`.
- Gradient infinity norm was finite: `4.2012939453125`.
- No `results.json` or `REJECTED.json` was produced.

Matched checkpoint timings regressed rather than improved:

- `initial_hardware_status_returned`: `131.31 s` patched vs `113.53 s` previous.
- `target_lane_initial_objective_value_and_grad_started`: `168.45 s` patched vs `148.82 s` previous.
- `target_lane_initial_objective_value_and_grad_returned`: `209.42 s` patched vs `187.49 s` previous.
- `phase1_attempt_0_started`: `209.44 s` patched vs `187.54 s` previous.

This is the decisive experiment. A backend-local fixed-trip optimization can improve synthetic loop microbenchmarks, but it does not solve SIMSOPT's dynamic convergence loop bottleneck.

## Why Small Fixes Are Not Enough

### Environment Flags

Flags can help measure or route execution. They cannot remove the fundamental host scalar-read boundary from dynamic convergence loops.

### Fixed-Trip Chunking

Fixed-trip chunking helps when the backend can prove a loop is equivalent to `i < constant`.

SIMSOPT's solver loops are not purely fixed-trip loops. They depend on residual norms, convergence status, and GMRES/Newton state. Chunking them blindly would change numerical semantics unless the loop is rewritten with masking and strong parity tests.

### Outer Optimizer Changes

Moving or tuning the host L-BFGS-B wrapper does not remove the inner solver-loop synchronization, because SciPy host control is shared by CPU and MPS lanes. The problematic work is inside the JAX objective-and-gradient evaluation.

### More JIT Caching

Caching can reduce repeated compile cost. It does not fix per-trip runtime synchronization once the compiled executable is running.

### CPU Fallback

CPU fallback avoids the MPS backend issue by not using MPS for the hot loop. It is a workaround, not an MPS acceleration solution.

## What A Real Fix Requires

There are three plausible project classes.

### Option A: jax-mps Backend Fix

Implement a real device-resident dynamic loop path for `stablehlo.while` in jax-mps/MLX.

The key requirement is that the convergence predicate must not be exposed as a host `.item<bool>()` every trip. The backend needs either:

- A true MLX/Metal primitive that owns the loop internally.
- A chunked dynamic loop primitive that evaluates multiple trips per host boundary while preserving exact semantics.
- A compiler/runtime lowering that can keep body, condition, and predicate branching in device-side execution.

This is backend work because it changes how jax-mps executes StableHLO control flow.

### Option B: SIMSOPT Custom Kernel Or Primitive

Build custom MLX/C++/Metal primitives for the Boozer residual, Newton update, GMRES matvec/solve segment, or a fused value-and-gradient region.

The goal is to present jax-mps with fewer, larger operations:

- One custom op for a solver block instead of hundreds or thousands of loop trips.
- Internal device-side convergence handling.
- Explicit host result only at the final objective/gradient boundary.

This is a custom-kernel project because the useful abstraction boundary is below Python/JAX expression rewrites.

### Option C: SIMSOPT Algorithm Restructuring

Rewrite the hot solver path into fixed-iteration, masked, batched, or more parallel forms.

This could make the workload friendlier to current jax-mps, but it is scientifically risky:

- Fixed-iteration masked Newton/GMRES must preserve convergence behavior.
- Any early-stop removal changes numerical work and possible failure modes.
- The result needs CPU/CUDA/MPS parity checks.

This is still a real implementation project, not a parameter tweak.

## Recommended Next Plan

1. Dump the StableHLO/MLIR for the cached strict scalar `value_and_grad` and classify every `while` as fixed-trip, bounded dynamic, or data-dependent dynamic.
2. Add a small GMRES-like dynamic convergence microbenchmark that matches SIMSOPT's loop structure better than the existing fixed-trip tests.
3. Prototype a backend dynamic-chunk primitive on that benchmark, with exact parity tests against CPU JAX.
4. If backend dynamic chunking cannot preserve semantics cleanly, move to a custom MLX extension for the Boozer/GMRES hot block.
5. Re-run the SIMSOPT `scipy-jax` MPS `maxiter=3` smoke and require an actual completed run before claiming improvement.
6. Only then compare CPU, CUDA, and MPS at matched problem sizes.

## Decision

The current best answer is:

> It is a real backend/custom-kernel project.

The reason is not that GPUs are inherently bad for this workload. CUDA evidence says a GPU can win. The reason is that current jax-mps turns SIMSOPT's dynamic solver control flow into repeated host/device synchronization, and the attempted fixed-trip backend optimization improved microbenchmarks but did not move the real SIMSOPT run.

The next credible work is backend-level dynamic control-flow execution or a custom MLX/SIMSOPT kernel boundary. Anything smaller should be treated as a diagnostic or workaround until it completes the matched SIMSOPT smoke run.
